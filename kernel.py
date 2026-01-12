import torch
import triton
import triton.language as tl

@triton.jit
def ssm_scan_kernel(
    cumsum_A_ptr, x_ptr, out_ptr,
    batch, seq, dim,
    stride_ca_b, stride_ca_s, stride_ca_d,
    stride_x_b, stride_x_s, stride_x_r, stride_x_c,
    stride_o_b, stride_o_s, stride_o_r, stride_o_c,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    SSM parallel scan kernel - avoids materializing the O(seq^2) decay matrix.
    
    The key idea: each GPU thread handles one (batch, timestep, row) triple and 
    computes the weighted sum over all previous timesteps for that row.
    
    This gives us out[b,t,r,c] = sum_{j=0}^{t} exp(A[b,t,r] - A[b,j,r]) * x[b,j,r,c]
    """
    pid = tl.program_id(axis=0)
    
    # Figure out which (batch, time, row) this thread is handling
    total_rows = batch * seq * dim
    if pid >= total_rows:
        return
    
    # Unpack the linearized index
    r = pid % dim
    pid_temp = pid // dim
    t = pid_temp % seq
    b = pid_temp // seq
    
    # Cache the cumsum value for this output position
    cumsum_t_r = tl.load(cumsum_A_ptr + b * stride_ca_b + t * stride_ca_s + r * stride_ca_d)
    
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    
    # Process columns in blocks (dim might be larger than BLOCK_SIZE_C)
    for c_start in range(0, dim, BLOCK_SIZE_C):
        c_idx = c_start + c_offsets
        c_mask = c_idx < dim
        
        acc = tl.zeros([BLOCK_SIZE_C], dtype=tl.float32)
        
        # Sum over all previous timesteps (causal masking)
        for j in range(t + 1):
            cumsum_j_r = tl.load(cumsum_A_ptr + b * stride_ca_b + j * stride_ca_s + r * stride_ca_d)
            
            # Decay factor from timestep j to t
            decay = tl.exp(cumsum_t_r - cumsum_j_r)
            
            # Load the input values and accumulate
            x_offsets = b * stride_x_b + j * stride_x_s + r * stride_x_r + c_idx * stride_x_c
            x_vals = tl.load(x_ptr + x_offsets, mask=c_mask, other=0.0)
            
            acc += decay * x_vals
        
        # Write out the result
        out_offsets = b * stride_o_b + t * stride_o_s + r * stride_o_r + c_idx * stride_o_c
        tl.store(out_ptr + out_offsets, acc, mask=c_mask)


def ssm_scan_triton(cumsum_A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Fast SSM scan using Triton - computes the parallel scan without 
    materializing the massive decay matrix.
    
    Given:
        cumsum_A: (batch, seq, dim) cumulative log decay 
        x: (batch, seq, dim, dim) input sequence
    
    Returns:
        out: (batch, seq, dim, dim) where 
             out[b,t] = sum_{j<=t} exp(A[b,t] - A[b,j]) * x[b,j]
    """
    batch, seq, dim = cumsum_A.shape
    assert x.shape == (batch, seq, dim, dim)
    
    out = torch.empty_like(x)
    
    # One thread block per (batch, timestep, row) - that's our parallelization strategy
    grid = lambda meta: (batch * seq * dim,)
    
    # Block size for vectorizing over columns (needs to be power of 2)
    BLOCK_SIZE_C = triton.next_power_of_2(dim)
    BLOCK_SIZE_C = min(BLOCK_SIZE_C, 128)  # don't go too crazy with registers
    
    ssm_scan_kernel[grid](
        cumsum_A, x, out,
        batch, seq, dim,
        cumsum_A.stride(0), cumsum_A.stride(1), cumsum_A.stride(2),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return out