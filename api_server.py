"""
Lemonade Bridge — OpenAI-Compatible API Server

Exposes the Fluidic-Hybrid Backbone through OpenAI chat completions API.
Drop-in compatible with LibreChat, Open WebUI, Continue.dev, or any client
that speaks the ChatCompletions protocol.

Usage:
    python api_server.py

Endpoints:
    GET  /v1/models              — List available models
    POST /v1/chat/completions     — Generate text (streaming supported)
    GET  /health                  — Health check
"""
from fastapi.middleware.cors import CORSMiddleware
import torch
import time
import uuid
import json
from tokenizers import Tokenizer
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from model import FluidicHybridBackbone

# ============================================================================
# Model Configuration
# ============================================================================
VOCAB_SIZE = 16384
TOKENIZER_PATH = "data/tokenizer.json"
MAX_NEW_TOKENS = 128

MODELS_CONFIG = {
    "fluidic-hybrid-10.5M": {
        "D_MODEL": 256,
        "STATE_DIM": 256,
        "CFC_HIDDEN": 256,
        "NUM_LAYERS": 4,
        "CHECKPOINT_PATH": "models/v1_10.5M/final.pt"
    },
    "fluidic-hybrid-17.3M": {
        "D_MODEL": 384,
        "STATE_DIM": 384,
        "CFC_HIDDEN": 384,
        "NUM_LAYERS": 4,
        "CHECKPOINT_PATH": "models/v2_17.3M/final.pt"
    }
}

# ============================================================================
# Global Model State
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_registry = {}
tokenizer = None

def load_models():
    global models_registry, tokenizer
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"[Lemonade Bridge] Initialization started on {device}...")
    
    for model_id, config in MODELS_CONFIG.items():
        try:
            model = FluidicHybridBackbone(
                vocab_size=VOCAB_SIZE,
                d_model=config["D_MODEL"],
                state_dim=config["STATE_DIM"],
                cfc_hidden_size=config["CFC_HIDDEN"],
                num_layers=config["NUM_LAYERS"]
            ).to(device)
            
            model.load_state_dict(torch.load(config["CHECKPOINT_PATH"], map_location=device, weights_only=True))
            model.eval()
            
            models_registry[model_id] = model
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  --> Loaded {model_id} ({params:.2f}M params) from {config['CHECKPOINT_PATH']}")
        except Exception as e:
            print(f"  --> Failed to load {model_id}: {e}")
            
    print(f"[Lemonade Bridge] All models online. Total VRAM: {torch.cuda.max_memory_allocated()/(1024**2):.0f}MB")

# ============================================================================
# OpenAI-Compatible API Schema
# ============================================================================
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "fluidic-hybrid-17.3M"
    messages: List[Message]
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    temperature: Optional[float] = 0.8
    stream: Optional[bool] = False

# ============================================================================
# FastAPI Application
# ============================================================================
app = FastAPI(title="Fluidic-Hybrid Lemonade Bridge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    load_models()

@app.get("/api/v1/models")
async def list_models():
    """OpenAI-compatible model listing endpoint."""
    available_models = []
    for model_id in models_registry.keys():
        available_models.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nolimit-fluidic",
            "permission": [],
            "root": model_id,
            "parent": None,
        })
        
    return {
        "object": "list",
        "data": available_models
    }

@app.post("/api/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint using continuous logic."""
    
    requested_model_id = request.model
    if requested_model_id not in models_registry:
        # Fallback to the default 17.3M if unknown model requested
        requested_model_id = "fluidic-hybrid-17.3M"
        if requested_model_id not in models_registry:
             # Fallback to first available model if 17.3M failed to load
             requested_model_id = list(models_registry.keys())[0]
             
    model_instance = models_registry[requested_model_id]
    config = MODELS_CONFIG[requested_model_id]
    
    # Combine all messages into a single prompt
    prompt = " ".join([m.content for m in request.messages])
    
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Initialize persistent continuous states using the requested model's dims
    ssm_states = [torch.zeros(1, config["STATE_DIM"]).to(device) for _ in range(config["NUM_LAYERS"])]
    cfc_states = [torch.zeros(1, config["CFC_HIDDEN"]).to(device) for _ in range(config["NUM_LAYERS"])]
    
    generated_tokens = []
    
    with torch.no_grad():
        # Process the prompt through the continuous logic engine
        logits, ssm_states, cfc_states = model_instance(input_tensor, ssm_states, cfc_states)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # Sample from the last position
        next_logits = logits[0, -1, :] / max(request.temperature, 0.01)
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated_tokens.append(next_token)
        
        # Auto-regressive continuous generation
        for _ in range(request.max_tokens - 1):
            current_input = torch.tensor([[next_token]], dtype=torch.long).to(device)
            logits, ssm_states, cfc_states = model_instance(current_input, ssm_states, cfc_states)
            
            next_logits = logits[0, -1, :] / max(request.temperature, 0.01)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Stop on EOS token (token ID 3)
            if next_token == 3:
                break
                
            generated_tokens.append(next_token)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tps = len(generated_tokens) / elapsed if elapsed > 0 else 0
        vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        print(f"[Lemonade Bridge] Request completed: {len(generated_tokens)} tokens @ {tps:.1f} tok/s | Peak VRAM: {vram:.0f} MB")
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    
    # Return OpenAI-compatible response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": "stop" if next_token == 3 else "length"
        }],
        "usage": {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(generated_tokens),
            "total_tokens": len(input_ids) + len(generated_tokens)
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "models": list(models_registry.keys()), "device": str(device)}

# ============================================================================
# Entry Point
# ============================================================================
import socket

def find_available_port(start_port=8000, max_port=8100):
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('0.0.0.0', port)) != 0:
                return port
    raise RuntimeError("No available ports found.")

if __name__ == "__main__":
    port = find_available_port()
    print("=" * 60)
    print("  Fluidic-Hybrid Backbone — Lemonade-Compatible API Server")
    print(f"  OpenAI-compatible endpoint: http://localhost:8001")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)
