"""
Fluidic-Hybrid Backbone — Interactive Test Script

Test any checkpoint with custom prompts. Works while training is running.

Usage:
    python test.py                              # Latest checkpoint, default prompts
    python test.py --checkpoint models/fluidic_ckpt_300.pt
    python test.py --prompt "Once upon a time"
    python test.py --tokens 50 --temperature 1.0
"""

import torch
import argparse
import glob
import os
import time
from tokenizers import Tokenizer
from model import FluidicHybridBackbone


def find_latest_checkpoint():
    """Find the most recent checkpoint in models/."""
    checkpoints = glob.glob("models/*/checkpoints/iter_*.pt") + glob.glob("models/*/final.pt")
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def generate(model, tokenizer, prompt, device, num_tokens=30, temperature=0.8, top_k=40):
    """Autoregressive generation with temperature and top-k sampling. Returns (text, tokens_per_sec)."""
    ids = tokenizer.encode(prompt).ids
    inp = torch.tensor([ids], dtype=torch.long).to(device)
    tokens = []

    with torch.no_grad():
        logits, ssm, cfc = model(inp)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(num_tokens):
            scaled = logits[0, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                vals, idx = torch.topk(scaled, top_k)
                scaled = torch.full_like(scaled, float('-inf'))
                scaled.scatter_(0, idx, vals)
            probs = torch.softmax(scaled, dim=-1)
            tok = torch.multinomial(probs, 1).item()
            tokens.append(tok)
            logits, ssm, cfc = model(
                torch.tensor([[tok]], dtype=torch.long).to(device), ssm, cfc
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tps = len(tokens) / elapsed if elapsed > 0 else 0

    return tokenizer.decode(tokens), tps


def main():
    parser = argparse.ArgumentParser(description="Test Fluidic-Hybrid model checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default: latest)")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--tokens", type=int, default=30, help="Tokens to generate (default: 30)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering (default: 40)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("data/tokenizer.json")

    ckpt = args.checkpoint or find_latest_checkpoint()
    if not ckpt or not os.path.exists(ckpt):
        print("No checkpoint found. Run train.py first.")
        return

    model = FluidicHybridBackbone(16384, 384, 384, 384, 4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded: {ckpt} ({params:.1f}M params)")
    print(f"Settings: tokens={args.tokens}, temp={args.temperature}, top_k={args.top_k}\n")

    prompts = [args.prompt] if args.prompt else [
        "The history of",
        "In the beginning",
        "Science has shown that",
        "The city was built",
        "During the war",
        "Hello thee how are you"
    ]

    all_tps = []
    for prompt in prompts:
        output, tps = generate(model, tokenizer, prompt, device, args.tokens, args.temperature, args.top_k)
        all_tps.append(tps)
        print(f"[{prompt}] → {output}")
        print(f"  ({tps:.1f} tokens/sec)\n")

    avg_tps = sum(all_tps) / len(all_tps)
    print(f"{'='*50}")
    print(f"Average: {avg_tps:.1f} tokens/sec")
    if torch.cuda.is_available():
        print(f"VRAM:    {torch.cuda.max_memory_allocated()/(1024**2):.0f} MB")


if __name__ == "__main__":
    main()
