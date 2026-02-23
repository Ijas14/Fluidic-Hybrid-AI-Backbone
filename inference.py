"""
Fluidic-Hybrid AI Backbone — Inference & Text Generation

Loads a trained checkpoint and generates text using autoregressive
sampling with persistent S4/CfC states and configurable temperature.
"""

import torch
from tokenizers import Tokenizer
import os

from model import FluidicHybridBackbone


def generate(model, tokenizer, prompt, device, num_tokens=50, temperature=0.8, top_k=40):
    """
    Autoregressive text generation with temperature and top-k sampling.

    Args:
        model: Trained FluidicHybridBackbone
        tokenizer: BPE tokenizer
        prompt: Input text string
        device: torch.device
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k filtering (0 = greedy)
    Returns:
        Generated text string
    """
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    ssm_states = None
    cfc_states = None
    generated_tokens = []

    with torch.no_grad():
        # Process prompt to build context state
        logits, ssm_states, cfc_states = model(input_tensor, ssm_states, cfc_states)

        for _ in range(num_tokens):
            # Sample from last position
            next_logits = logits[0, -1, :] / max(temperature, 1e-8)

            if top_k > 0:
                values, indices = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(0, indices, values)

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_tokens.append(next_token)

            # Feed single token with persistent states
            current_input = torch.tensor([[next_token]], dtype=torch.long).to(device)
            logits, ssm_states, cfc_states = model(current_input, ssm_states, cfc_states)

    return tokenizer.decode(generated_tokens)


def main():
    """Load checkpoint and run inference on sample prompts."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer_path = "data/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Model configuration (must match train.py)
    VOCAB_SIZE = 16384
    D_MODEL = 384
    STATE_DIM = 384
    CFC_HIDDEN = 384
    NUM_LAYERS = 4

    model = FluidicHybridBackbone(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        state_dim=STATE_DIM,
        cfc_hidden_size=CFC_HIDDEN,
        num_layers=NUM_LAYERS
    ).to(device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {params:.2f}M parameters")

    # Load checkpoint
    checkpoint_file = "models/v2_17.3M/final.pt"
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}. Run train.py first.")

    print(f"Loading weights: {checkpoint_file}")
    model.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))
    print("Weights loaded successfully.\n")

    # Generate from sample prompts
    prompts = [
        "The history of",
        "In the beginning",
        "Science has shown that",
    ]

    for prompt in prompts:
        output = generate(model, tokenizer, prompt, device, num_tokens=30)
        print(f"[Prompt]: {prompt}")
        print(f"[Output]: {output}")
        print("-" * 60)

    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\nPeak VRAM: {vram:.0f} MB")


if __name__ == "__main__":
    main()
