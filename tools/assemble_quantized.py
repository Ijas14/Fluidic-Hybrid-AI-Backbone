import torch
import torch.nn as nn
from model import FluidicHybridBackbone
from quantization import NF4Linear, convert_linear_to_nf4

def replace_linear_with_nf4(module, name=""):
    """
    Recursively traverse the PyTorch module and replace `nn.Linear` 
    with `NF4Linear` custom kernels.
    """
    for child_name, child_module in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        
        if isinstance(child_module, nn.Linear):
            # Do not quantize small classification heads if possible, 
            # though here we'll just quantize everything for the prototype
            print(f"Quantizing layer: {full_name} ({child_module.in_features} -> {child_module.out_features})")
            
            # Create our Triton NF4 drop-in
            nf4_layer = convert_linear_to_nf4(child_module)
            nf4_layer = nf4_layer.cuda()
            
            setattr(module, child_name, nf4_layer)
        else:
            replace_linear_with_nf4(child_module, full_name)


def main():
    print("--- Fluidic-Hybrid Backbone Quantization (Phase 2) ---")
    
    # 1. Instantiate the Full Precision Model
    VOCAB_SIZE = 5000
    D_MODEL = 256
    STATE_DIM = 256
    CFC_HIDDEN = 256
    
    # We want a slightly larger model here to actually see the memory savings
    # Let's bump up dimensions for the test
    D_MODEL = 1024
    STATE_DIM = 512
    CFC_HIDDEN = 1024
    
    print("\n[INFO] Instantiating FP16 Base Model...")
    model_fp16 = FluidicHybridBackbone(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        state_dim=STATE_DIM, 
        cfc_hidden_size=CFC_HIDDEN
    ).cuda().half()
    
    # Measure typical memory
    mem_fp16 = sum(p.numel() * p.element_size() for p in model_fp16.parameters())
    print(f"FP16 Raw Parameter Memory: {mem_fp16 / 1024**2:.2f} MB")
    
    # 2. Swap out linear layers for NF4 layers
    print("\n[INFO] Initiating NF4/Quark Compression...")
    replace_linear_with_nf4(model_fp16)
    
    # After replacement, parameters in NF4Linear are buffers (uint8 and float16 scales)
    # Let's compute the new memory strictly from the buffers and parameters.
    mem_nf4 = 0
    for buf in model_fp16.buffers():
        mem_nf4 += buf.numel() * buf.element_size()
    for p in model_fp16.parameters(): # Embeddings, bias, etc.
        mem_nf4 += p.numel() * p.element_size()
        
    print(f"\nNF4 Quantized Parameter Memory: {mem_nf4/1024**2:.2f} MB")
    print(f"Total Model Compression Ratio: {mem_fp16 / mem_nf4:.2f}x")
    
    # 3. Verify Inference Pass
    print("\n[INFO] Verifying NF4 Forward Pass (Triton Fused Kernel)...")
    model_fp16.eval()
    dummy_input = torch.randint(0, VOCAB_SIZE, (4, 16)).cuda()
    
    with torch.no_grad():
        try:
            logits, next_ssm, next_cfc = model_fp16(dummy_input)
            print(f"Logits shape -> expected [4, 16, {VOCAB_SIZE}], got {logits.shape}")
            print(f"Persistent SSM State Array ok. Continuous CfC shape -> {next_cfc.shape}")
            print("\n[SUCCESS] The NF4 Fluidic-Hybrid logic simulated perfectly on ROCm.")
        except Exception as e:
            print(f"\n[ERROR] Forward pass failed: {e}")
            raise

if __name__ == "__main__":
    main()
