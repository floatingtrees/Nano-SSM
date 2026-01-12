import torch
import triton
import triton.language as tl

@triton.jit  
def sequential_scan_kernel(
    log_A_ptr, x_ptr, out_ptr, batch, seq, dim,
    stride_a_b, stride_a_s, stride_a_d,
    stride_x_b, stride_x_s, stride_x_r, stride_x_c,
    stride_o_b, stride_o_s, stride_o_r, stride_o_c,
):
    """
    O(n) work and time, beats naive python
    """
    pid = tl.program_id(axis=0)
    
    total_elements = batch * dim * dim
    if pid >= total_elements:
        return
    
    c = pid % dim
    pid_temp = pid // dim
    r = pid_temp % dim
    b = pid_temp // dim
    
    y = 0.0 
    
    for t in range(seq):
        # Load log_A[b, t, r] and x[b, t, r, c]
        log_a = tl.load(log_A_ptr + b * stride_a_b + t * stride_a_s + r * stride_a_d)
        x_val = tl.load(x_ptr + b * stride_x_b + t * stride_x_s + r * stride_x_r + c * stride_x_c)
        
        # Apply recurrence: y[t] = A[t] * y[t-1] + x[t]
        # For t=0, there's no previous state, so y = x
        if t == 0:
            y = x_val
        else:
            a = tl.exp(log_a)
            y = a * y + x_val
        
        # Store output
        tl.store(out_ptr + b * stride_o_b + t * stride_o_s + r * stride_o_r + c * stride_o_c, y)


def ssm_forward_triton(log_A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    O(n) SSM forward pass
    
    Computes y[t] = A[t] * y[t-1] + x[t]
    
    Args:
        log_A: (batch, seq, dim) - log of diagonal decay values
        x: (batch, seq, dim, dim) - input sequence
    
    Returns:
        out: (batch, seq, dim, dim) - output sequence
    """
    batch, seq, dim = log_A.shape
    assert x.shape == (batch, seq, dim, dim)
    
    out = torch.empty_like(x)
    
    grid = (batch * dim * dim,)
    
    sequential_scan_kernel[grid](
        log_A, x, out, batch, seq, dim,
        log_A.stride(0), log_A.stride(1), log_A.stride(2),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    
    return out

