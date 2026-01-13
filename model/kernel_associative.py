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
    O(n) forward pass: y[t] = A[t] * y[t-1] + x[t]
    Supports both fp32 and fp64 (dtype inferred from input).
    """
    pid = tl.program_id(axis=0)
    
    total_elements = batch * dim * dim
    if pid >= total_elements:
        return
    
    c = pid % dim
    pid_temp = pid // dim
    r = pid_temp % dim
    b = pid_temp // dim
    
    # Load first x value to initialize y with correct dtype
    y = tl.load(x_ptr + b * stride_x_b + 0 * stride_x_s + r * stride_x_r + c * stride_x_c)
    tl.store(out_ptr + b * stride_o_b + 0 * stride_o_s + r * stride_o_r + c * stride_o_c, y)
    
    for t in range(1, seq):
        log_a = tl.load(log_A_ptr + b * stride_a_b + t * stride_a_s + r * stride_a_d)
        x_val = tl.load(x_ptr + b * stride_x_b + t * stride_x_s + r * stride_x_r + c * stride_x_c)
        
        a = tl.exp(log_a)
        y = a * y + x_val
        
        tl.store(out_ptr + b * stride_o_b + t * stride_o_s + r * stride_o_r + c * stride_o_c, y)


@triton.jit
def sequential_scan_backward_dx_kernel(
    log_A_ptr, dout_ptr, dx_ptr,
    batch, seq, dim,
    stride_a_b, stride_a_s, stride_a_d,
    stride_do_b, stride_do_s, stride_do_r, stride_do_c,
    stride_dx_b, stride_dx_s, stride_dx_r, stride_dx_c,
):
    """
    O(n) backward pass for dx: dx[t] = dout[t] + A[t+1] * dx[t+1]
    Supports both fp32 and fp64 (dtype inferred from input).
    """
    pid = tl.program_id(axis=0)
    
    total_elements = batch * dim * dim
    if pid >= total_elements:
        return
    
    c = pid % dim
    pid_temp = pid // dim
    r = pid_temp % dim
    b = pid_temp // dim
    
    # Start from last timestep - load to get correct dtype
    t = seq - 1
    dx_acc = tl.load(dout_ptr + b * stride_do_b + t * stride_do_s + r * stride_do_r + c * stride_do_c)
    tl.store(dx_ptr + b * stride_dx_b + t * stride_dx_s + r * stride_dx_r + c * stride_dx_c, dx_acc)
    
    # Process remaining timesteps in reverse
    for t_rev in range(1, seq):
        t = seq - 1 - t_rev
        
        dout_val = tl.load(dout_ptr + b * stride_do_b + t * stride_do_s + r * stride_do_r + c * stride_do_c)
        log_a_next = tl.load(log_A_ptr + b * stride_a_b + (t + 1) * stride_a_s + r * stride_a_d)
        a_next = tl.exp(log_a_next)
        dx_acc = dout_val + a_next * dx_acc
        
        tl.store(dx_ptr + b * stride_dx_b + t * stride_dx_s + r * stride_dx_r + c * stride_dx_c, dx_acc)


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


def ssm_backward_triton(log_A: torch.Tensor, out: torch.Tensor, dout: torch.Tensor) -> tuple:
    """
    O(n) SSM backward pass
    
    Args:
        log_A: (batch, seq, dim) - log of diagonal decay values from forward
        out: (batch, seq, dim, dim) - outputs from forward pass  
        dout: (batch, seq, dim, dim) - gradient of loss w.r.t. output
    
    Returns:
        dx: (batch, seq, dim, dim) - gradient w.r.t. x
        dlog_A: (batch, seq, dim) - gradient w.r.t. log_A
    """
    batch, seq, dim = log_A.shape
    
    dx = torch.empty_like(dout)
    
    grid = (batch * dim * dim,)
    
    # Step 1: Compute dx using triton kernel (fast, no atomics)
    sequential_scan_backward_dx_kernel[grid](
        log_A, dout, dx,
        batch, seq, dim,
        log_A.stride(0), log_A.stride(1), log_A.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
    )
    
    # Step 2: Compute dlog_A using PyTorch (optimized, avoids atomics)
    # dlog_A[b, t, r] = A[b, t, r] * sum_c(y[b, t-1, r, c] * dx[b, t, r, c]) for t > 0
    # dlog_A[b, 0, r] = 0 (A[0] not used in forward)
    
    A = torch.exp(log_A)  # (batch, seq, dim)
    
    # y_prev[t] = out[t-1] for t>0, zeros for t=0
    y_prev = torch.zeros_like(out)
    y_prev[:, 1:, :, :] = out[:, :-1, :, :]
    
    # (y_prev * dx).sum(dim=-1) gives (batch, seq, dim)
    dlog_A = A * (y_prev * dx).sum(dim=-1)
    
    return dx, dlog_A


class SSMFunction(torch.autograd.Function):
    """Differentiable SSM operator with O(n) forward and backward passes."""
    
    @staticmethod
    def forward(ctx, log_A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = ssm_forward_triton(log_A, x)
        ctx.save_for_backward(log_A, out)
        return out
    
    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        log_A, out = ctx.saved_tensors
        dx, dlog_A = ssm_backward_triton(log_A, out, dout)
        return dlog_A, dx


def ssm_scan(log_A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable O(n) SSM scan operator.
    
    Args:
        log_A: (batch, seq, dim) - log of diagonal decay values
        x: (batch, seq, dim, dim) - input sequence
    
    Returns:
        out: (batch, seq, dim, dim) - output sequence
    """
    return SSMFunction.apply(log_A, x)

