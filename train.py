"""
Fluidic-Hybrid Backbone — Training Script

Trains the Fluidic-Hybrid language model using:
    - DEQ Phantom Gradient (permanent vanishing gradient fix)
    - AMP FP16 mixed precision
    - Cosine LR with linear warmup
    - torch.compile() JIT kernel fusion
    - Zero-copy memory-mapped dataset streaming
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time

from model import FluidicHybridBackbone


def get_memmap_batch(data_array, batch_size, seq_len, device):
    """Sample a random batch from the memory-mapped corpus (zero-copy from SSD)."""
    max_idx = len(data_array) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([torch.from_numpy((data_array[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data_array[i+1:i+seq_len+1]).astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train():
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Dataset ---
    dataset_file = "data/train_corpus_16k.bin"
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset {dataset_file} not found. Run tools/prepare_data.py first.")

    print(f"Loading memory-mapped dataset: {dataset_file}")
    data_array = np.memmap(dataset_file, dtype=np.uint16, mode='r')
    print(f"Dataset mounted. Tokens available: {len(data_array):,}")

    # --- Model Configuration ---
    VOCAB_SIZE = 16384
    D_MODEL = 384
    STATE_DIM = 384
    CFC_HIDDEN = 384
    NUM_LAYERS = 4

    # --- Training Configuration ---
    PHYSICAL_BATCH = 8
    GRADIENT_ACCUMULATION = 8   # Effective batch = 64
    SEQ_LEN = 128
    MAX_ITERS = 1000
    LR_MAX = 1e-3
    LR_MIN = 1e-5
    WARMUP_ITERS = 200

    # Enable TF32 for faster matmuls on supported hardware
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Build Model ---
    print(f"\nBuilding Fluidic-Hybrid Backbone...")
    model = FluidicHybridBackbone(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        state_dim=STATE_DIM,
        cfc_hidden_size=CFC_HIDDEN,
        num_layers=NUM_LAYERS
    ).to(device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    print(f"Training: DEQ Phantom Gradient + AMP FP16")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1)

    def get_lr(step):
        if step < WARMUP_ITERS:
            return LR_MAX * step / WARMUP_ITERS
        progress = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
        return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * progress))

    scaler = GradScaler()
    model.train()

    # --- Training Loop ---
    print(f"\nStarting training ({MAX_ITERS} iterations, batch={PHYSICAL_BATCH}x{GRADIENT_ACCUMULATION})...")
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for iter_idx in range(MAX_ITERS):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(GRADIENT_ACCUMULATION):
            X, Y = get_memmap_batch(data_array, PHYSICAL_BATCH, SEQ_LEN, device)

            with autocast(dtype=torch.float16):
                logits, _, _ = model(X)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), Y.view(-1))
                loss = loss / GRADIENT_ACCUMULATION

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        lr = get_lr(iter_idx)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_idx % 20 == 0:
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            elapsed = time.time() - start_time
            print(f"Iter {iter_idx:04d} | Loss: {accumulated_loss:.4f} | LR: {lr:.2e} | VRAM: {vram:.0f}MB | Time: {elapsed:.1f}s")

        if iter_idx % 100 == 0 and iter_idx > 0:
            ckpt_dir = "models/v2_17.3M/checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = f"{ckpt_dir}/iter_{iter_idx}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Checkpoint saved: {ckpt_path}")

    # Save final weights
    final_dir = "models/v2_17.3M"
    os.makedirs(final_dir, exist_ok=True)
    final_path = f"{final_dir}/final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final weights: {final_path}")


if __name__ == "__main__":
    train()
