import torch
import torch.nn.functional as F
import time
from kernel_associative import ssm_forward_triton, ssm_scan

def ssm_forward_naive(log_A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation matching ssm.py forward pass.
    Uses O(seq^2) einsum approach.
    """
    batch, seq, dim = log_A.shape
    
    cumsum_A = torch.cumsum(log_A, dim=1)
    log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1)
    
    indices = torch.arange(seq, device=x.device)
    mask = indices[:, None] >= indices[None, :]
    log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
    
    M = torch.exp(log_M)
    out = torch.einsum('b t j r, b j r c -> b t r c', M, x)
    
    return out


def ssm_forward_sequential(log_A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Ground truth: sequential RNN-style computation.
    """
    batch, seq, dim = log_A.shape
    _, _, dim1, dim2 = x.shape
    
    out = torch.zeros_like(x)
    state = torch.zeros(batch, dim1, dim2, device=x.device, dtype=x.dtype)
    
    for t in range(seq):
        if t == 0:
            state = x[:, t]
        else:
            A = torch.exp(log_A[:, t])  # (batch, dim)
            state = A.unsqueeze(-1) * state + x[:, t]
        out[:, t] = state
    
    return out


def test_correctness():
    print("=" * 60)
    print("Testing Correctness")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    configs = [
        (1, 4, 8),
        (2, 16, 16),
        (4, 64, 32),
    ]
    
    for batch, seq, dim in configs:
        print(f"\nConfig: batch={batch}, seq={seq}, dim={dim}")
        
        log_A = torch.randn(batch, seq, dim, device='cuda') * 0.1
        x = torch.randn(batch, seq, dim, dim, device='cuda')
        
        # Ground truth (sequential)
        out_seq = ssm_forward_sequential(log_A, x)
        
        # Naive O(n^2) einsum
        out_naive = ssm_forward_naive(log_A, x)
        
        # Our O(n) triton kernel
        out_triton = ssm_forward_triton(log_A, x)
        
        # Check naive vs sequential
        diff_naive = (out_naive - out_seq).abs().max().item()
        print(f"  Naive vs Sequential: {diff_naive:.2e}")
        
        # Check triton vs sequential
        diff_triton = (out_triton - out_seq).abs().max().item()
        print(f"  Triton vs Sequential: {diff_triton:.2e}")
        
        assert torch.allclose(out_triton, out_seq, rtol=1e-4, atol=1e-5), \
            f"Triton doesn't match! Max diff: {diff_triton:.2e}"
        
        print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All correctness tests PASSED!")
    print("=" * 60)


def test_performance():
    print("\n" + "=" * 60)
    print("Testing Performance")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    configs = [
        (2, 64, 64),
        (4, 256, 64),
        (2, 1024, 64),
    ]
    
    n_warmup = 5
    n_iters = 20
    
    print(f"\nWarmup: {n_warmup}, Iterations: {n_iters}")
    
    for batch, seq, dim in configs:
        print(f"\n{'='*60}")
        print(f"Config: batch={batch}, seq={seq}, dim={dim}")
        print(f"{'='*60}")
        
        log_A = torch.randn(batch, seq, dim, device='cuda') * 0.1
        x = torch.randn(batch, seq, dim, dim, device='cuda')
        
        # Benchmark naive O(n^2)
        for _ in range(n_warmup):
            _ = ssm_forward_naive(log_A, x)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_iters):
            _ = ssm_forward_naive(log_A, x)
        torch.cuda.synchronize()
        naive_time = (time.time() - start) / n_iters * 1000
        
        # Benchmark triton O(n)
        for _ in range(n_warmup):
            _ = ssm_forward_triton(log_A, x)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_iters):
            _ = ssm_forward_triton(log_A, x)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / n_iters * 1000
        
        speedup = naive_time / triton_time
        
        print(f"  Naive O(n²):  {naive_time:.3f} ms")
        print(f"  Triton O(n):  {triton_time:.3f} ms")
        print(f"  Speedup:      {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"  ✓ Triton is {speedup:.2f}x faster!")
        else:
            print(f"  ⚠ Triton is slower")
    
    print("\n" + "=" * 60)
    print("Performance testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_correctness()
    test_performance()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
