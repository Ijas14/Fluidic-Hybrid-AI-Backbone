import os
import numpy as np
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def build_tokenizer_and_dataset():
    print("--- Phase 5: Building Custom 16k BPE Tokenizer & Zero-Copy Dataset ---")
    
    # 1. Load Dataset
    print("\nLoading Wikitext dataset...")
    try:
        # Try loading wikitext-103
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    except Exception as e:
        print(f"Failed to load wikitext-103. Falling back to wikitext-2 for speed. {e}")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
    print(f"Dataset lines: {len(dataset)}")
    
    # 2. Train Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=16384)
    
    # We train on a subset to save RAM and time
    def batch_iterator():
        for i in range(0, len(dataset), 1000):
            yield dataset[i:i + 1000]["text"]
            if i > 50000: # Limit training tokens
                break
                
    print("\nTraining 16k Custom BPE Tokenizer (RAM Optimized)...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save("fluidic_bpe_16k.json")
    print("Tokenizer successfully compiled and saved to 'fluidic_bpe_16k.json'")
    
    # 3. Compile Dataset to memory-mapped binary file
    print("\nTokenizing corpus and building numpy.memmap binary...")
    
    # To protect 8GB RAM, we process and write in chunks
    bin_file = "train_corpus_16k.bin"
    
    # Determine type based on vocab size -> uint16 is enough for 16,384 (max 65535)
    dtype = np.uint16
    
    # Process in batches and stream strictly to SSD
    print("Writing tokens to disk (RAM Safe Chunking)...")
    total_tokens = 0
    with open(bin_file, "wb") as f:
        batch_size = 5000
        # For prototype speed, we'll limit to 500,000 lines. Can expand later.
        for i in range(0, min(len(dataset), 500000), batch_size): 
            batch_texts = dataset[i:i+batch_size]["text"]
            # Encode batch
            encodings = tokenizer.encode_batch(batch_texts)
            # Flatten
            batch_ids = []
            for enc in encodings:
                if len(enc.ids) > 0:
                    batch_ids.extend(enc.ids)
                    batch_ids.append(3) # [EOS] = 3
                    
            if len(batch_ids) > 0:
                arr = np.array(batch_ids, dtype=dtype)
                f.write(arr.tobytes())
                total_tokens += len(batch_ids)
                
            if i % 25000 == 0 and i > 0:
                print(f"   Processed {i} lines... ({total_tokens} tokens so far)")
                
    print(f"\nCompilation complete! Total tokens structured: {total_tokens}")
    print(f"Dataset compiled to {bin_file} ({os.path.getsize(bin_file) / 1024 / 1024:.2f} MB)")
    print("The 8GB RAM Zero-Copy infrastructure is ready for the 25M execution loop.")

if __name__ == "__main__":
    build_tokenizer_and_dataset()
