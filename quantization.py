import torch
import torch.nn as nn
from core_modules import fused_nf4_dequant_matmul_kernel, NF4_QUANTS

class NF4Linear(nn.Module):
    """
    A PyTorch drop-in replacement for nn.Linear that stores weights in 4-bit NF4 format
    and uses the custom AMD-optimized Triton kernel for fused dequantization and matrix multiplication.
    """
    def __init__(self, in_features, out_features, bias=False, group_size=128):
        super(NF4Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # We store the quantized weights in INT8 (where each byte holds TWO 4-bit values).
        # This gives a massive 4x reduction in VRAM.
        # Shape is (in_features, out_features) mathematically, but physically it's packed.
        # For simplicity of triton, we store it transposed or physically packed by K.
        # K is in_features, N is out_features.
        # B_quant shape: (in_features, out_features // 2) if packed along N, 
        # but our triton prototype packed along K. 
        # Let's pack along the K dimension: (in_features // 2, out_features)
        
        # Initialization
        self.register_buffer(
            "weight_quant", 
            torch.zeros((in_features // 2, out_features), dtype=torch.uint8)
        )
        self.register_buffer(
            "weight_scales", 
            torch.ones((in_features // group_size, out_features), dtype=torch.float16)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
            
        # The lookup table tensor
        self.register_buffer(
            "nf4_lut", 
            torch.tensor(NF4_QUANTS, dtype=torch.float32)
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, in_features) or (M, K)
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            # Flatten to 2D for the Triton GEMM
            x = x.view(-1, self.in_features)
            
        M, K = x.shape
        N = self.out_features
        
        # Ensure memory is contiguous and precise for the kernel
        x = x.contiguous().to(torch.float16)
        
        # Preallocate output
        c = torch.empty((M, N), device=x.device, dtype=torch.float16)
        
        # Grid definition
        import triton
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        
        # Launch Fused Kernel
        fused_nf4_dequant_matmul_kernel[grid](
            x, self.weight_quant, self.weight_scales, c, self.nf4_lut,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight_quant.stride(0), self.weight_quant.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8
        )
        
        if self.bias is not None:
            c += self.bias
            
        if len(original_shape) == 3:
            c = c.view(original_shape[0], original_shape[1], -1)
            
        return c

def convert_linear_to_nf4(module: nn.Linear):
    """
    Mock utility to simulate taking a standard PyTorch Linear layer, compressing it
    using the Quark methodology, and returning an NF4Linear drop-in replacement.
    """
    in_features = module.in_features
    out_features = module.out_features
    bias_present = module.bias is not None
    
    nf4_layer = NF4Linear(in_features, out_features, bias=bias_present)
    
    # In a full AMD Quark integration, we would read the quantized tensors directly
    # from the `ModelQuantizer` output. For this architecture verification, 
    # we simulate the packaging to verify the Triton kernel's runtime shape logic.
    
    if bias_present:
        nf4_layer.bias.data.copy_(module.bias.data.to(torch.float16))
        
    return nf4_layer

if __name__ == "__main__":
    print("Testing NF4Linear py object...")
    layer = NF4Linear(256, 128).cuda()
    dummy_x = torch.randn(4, 10, 256, dtype=torch.float16).cuda()
    out = layer(dummy_x)
    print(f"Output shape: {out.shape} (Expected: [4, 10, 128])")
    print("NF4Linear instantiation and forward pass compiled successfully.")
