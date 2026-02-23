"""
Fluidic-Hybrid Backbone — Knowledge Distillation from DeepSeek-R1

Distills knowledge from DeepSeek-R1-1.5B (teacher) into the NOLIMIT
Fluidic-Hybrid model (student) using offline logit caching.

Pipeline:
    1. Generate teacher logits → save to disk (runs DeepSeek once)
    2. Train student against cached teacher logits (KL-divergence)

Usage:
    # Step 1: Generate teacher logits (run once, ~2-3 hours)
    python distill.py --generate-logits

    # Step 2: Distill into student (run after step 1, ~5 hours)
    python distill.py --train
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time
import argparse

from model import FluidicHybridBackbone


# =============================================================================
# Configuration
# =============================================================================
TEACHER_PATH = "DeepSeek-R1-1.5B-HF"
DATASET_FILE = "data/train_corpus_16k.bin"
TOKENIZER_PATH = "data/tokenizer.json"
LOGITS_CACHE_DIR = "data/teacher_logits"

# Student config (must match train.py)
VOCAB_SIZE = 16384
D_MODEL = 384
STATE_DIM = 384
CFC_HIDDEN = 384
NUM_LAYERS = 4

# Training config
SEQ_LEN = 128
PHYSICAL_BATCH = 4
GRADIENT_ACCUMULATION = 16  # Effective batch = 64
MAX_ITERS = 1000
LR_MAX = 5e-4
LR_MIN = 1e-5
WARMUP_ITERS = 200
TEMPERATURE = 2.0       # Distillation temperature (softens teacher logits)
ALPHA = 0.5             # Balance: 0 = pure distillation, 1 = pure CE


# =============================================================================
# Step 1: Generate Teacher Logits (Offline)
# =============================================================================
def generate_teacher_logits():
    """Run DeepSeek on the corpus and cache soft logits to disk."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(LOGITS_CACHE_DIR, exist_ok=True)

    print(f"Loading teacher model: {TEACHER_PATH}")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,  # NF4 quantization to fit in 4GB VRAM
    )
    teacher.eval()
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)
    print(f"Teacher loaded. VRAM: {torch.cuda.max_memory_allocated()/(1024**2):.0f}MB")

    # Load our tokenizer and dataset
    from tokenizers import Tokenizer
    student_tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    data_array = np.memmap(DATASET_FILE, dtype=np.uint16, mode='r')

    # Generate logits for N batches
    num_batches = MAX_ITERS * GRADIENT_ACCUMULATION
    print(f"Generating teacher logits for {num_batches} batches...")

    for batch_idx in range(num_batches):
        cache_file = os.path.join(LOGITS_CACHE_DIR, f"batch_{batch_idx:06d}.pt")
        if os.path.exists(cache_file):
            continue  # Skip already cached

        # Sample random sequences from corpus
        max_idx = len(data_array) - SEQ_LEN - 1
        ix = torch.randint(0, max_idx, (PHYSICAL_BATCH,))
        x = torch.stack([torch.from_numpy(data_array[i:i+SEQ_LEN].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data_array[i+1:i+SEQ_LEN+1].astype(np.int64)) for i in ix])

        # Decode student tokens → text → re-encode with teacher tokenizer
        texts = []
        for seq in x:
            text = student_tokenizer.decode(seq.tolist())
            texts.append(text)

        teacher_inputs = teacher_tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=SEQ_LEN
        ).to(device)

        with torch.no_grad():
            with autocast(dtype=torch.float16):
                teacher_out = teacher.forward(**teacher_inputs)

        # Save soft logits (top-k only to save disk space)
        logits = teacher_out.logits.float().cpu()
        top_k = 64
        top_vals, top_idx = torch.topk(logits, top_k, dim=-1)

        torch.save({
            "x": x,
            "y": y,
            "top_vals": top_vals.half(),
            "top_idx": top_idx.short(),
            "vocab_size": logits.shape[-1],
        }, cache_file)

        if batch_idx % 100 == 0:
            print(f"  Cached {batch_idx}/{num_batches} batches")

    print(f"Teacher logits saved to {LOGITS_CACHE_DIR}/")


# =============================================================================
# Step 2: Distillation Training
# =============================================================================
def distill_train():
    """Train student against cached teacher logits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(LOGITS_CACHE_DIR):
        raise FileNotFoundError(f"Run --generate-logits first. No cache at {LOGITS_CACHE_DIR}")

    # Load student model
    print("Building student model...")
    model = FluidicHybridBackbone(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL,
        state_dim=STATE_DIM, cfc_hidden_size=CFC_HIDDEN,
        num_layers=NUM_LAYERS
    ).to(device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Student: {params:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler()

    def get_lr(step):
        if step < WARMUP_ITERS:
            return LR_MAX * step / WARMUP_ITERS
        progress = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
        return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * progress))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model.train()

    print(f"\nStarting distillation (alpha={ALPHA}, T={TEMPERATURE})...")
    start_time = time.time()
    batch_counter = 0

    for iter_idx in range(MAX_ITERS):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(GRADIENT_ACCUMULATION):
            cache_file = os.path.join(LOGITS_CACHE_DIR, f"batch_{batch_counter:06d}.pt")
            if not os.path.exists(cache_file):
                break
            batch_counter += 1

            cached = torch.load(cache_file, weights_only=False)
            X = cached["x"].to(device)
            Y = cached["y"].to(device)

            with autocast(dtype=torch.float16):
                student_logits, _, _ = model(X)

                # Hard loss (standard CE)
                ce_loss = F.cross_entropy(
                    student_logits.view(-1, VOCAB_SIZE), Y.view(-1)
                )

                # Soft loss (KL-divergence with teacher)
                # Reconstruct teacher distribution from cached top-k
                teacher_logits = torch.full(
                    student_logits.shape, float('-inf'), device=device
                )
                top_vals = cached["top_vals"].float().to(device)
                top_idx = cached["top_idx"].long().to(device)

                # Only use matching sequence positions
                min_seq = min(student_logits.shape[1], top_vals.shape[1])
                min_vocab = min(VOCAB_SIZE, cached["vocab_size"])

                # Map teacher logits to student vocab space (top-k only)
                for b in range(X.shape[0]):
                    for t in range(min_seq):
                        valid_mask = top_idx[b, t] < VOCAB_SIZE
                        valid_idx = top_idx[b, t][valid_mask]
                        valid_vals = top_vals[b, t][valid_mask]
                        teacher_logits[b, t, valid_idx] = valid_vals

                teacher_probs = F.softmax(teacher_logits[:, :min_seq] / TEMPERATURE, dim=-1)
                student_log_probs = F.log_softmax(student_logits[:, :min_seq] / TEMPERATURE, dim=-1)
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                kl_loss = kl_loss * (TEMPERATURE ** 2)

                # Combined loss
                loss = ALPHA * ce_loss + (1 - ALPHA) * kl_loss
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
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            elapsed = time.time() - start_time
            print(f"Iter {iter_idx:04d} | Loss: {accumulated_loss:.4f} | LR: {lr:.2e} | VRAM: {vram:.0f}MB | Time: {elapsed:.1f}s")

        if iter_idx % 100 == 0 and iter_idx > 0:
            ckpt_dir = "models/distilled/checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = f"{ckpt_dir}/iter_{iter_idx}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  --> Checkpoint: {ckpt}")

    final_dir = "models/distilled"
    os.makedirs(final_dir, exist_ok=True)
    final = f"{final_dir}/final.pt"
    torch.save(model.state_dict(), final)
    print(f"\nDistillation complete. Weights: {final}")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill DeepSeek → NOLIMIT")
    parser.add_argument("--generate-logits", action="store_true", help="Step 1: Cache teacher logits")
    parser.add_argument("--train", action="store_true", help="Step 2: Distill into student")
    args = parser.parse_args()

    if args.generate_logits:
        generate_teacher_logits()
    elif args.train:
        distill_train()
    else:
        print("Usage:")
        print("  python distill.py --generate-logits  # Step 1: Cache teacher logits")
        print("  python distill.py --train            # Step 2: Train with distillation")
