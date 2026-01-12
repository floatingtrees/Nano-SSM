import torch
import time
from kernel import ssm_scan_triton

def ssm_scan_naive(cumsum_A, x):
    """
    Reference implementation - materializes the full O(seq^2) decay matrix.
    This matches the unoptimized version from ssm.py.
    """
    batch, seq, dim = cumsum_A.shape
    
    # Build the decay matrix: M[t,j] represents decay from time j to time t
    log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1)  # shape: (b, seq, seq, d)
    
    # Causal mask - can't attend to future
    indices = torch.arange(seq, device=x.device)
    mask = indices[:, None] >= indices[None, :]
    log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
    
    M = torch.exp(log_M)
    
    # Do the weighted sum via einsum
    out = torch.einsum('b t j r, b j r c -> b t r c', M, x)
    
    return out


def test_correctness():
    """Make sure the Triton kernel actually computes the right thing"""
    print("=" * 60)
    print("Testing Correctness")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    test_configs = [
        (1, 4, 8),     # small
        (2, 8, 16),    # medium
        (4, 16, 32),   # bigger
    ]
    
    for batch, seq, dim in test_configs:
        print(f"\nTesting batch={batch}, seq={seq}, dim={dim}")
        
        cumsum_A = torch.randn(batch, seq, dim, device='cuda', dtype=torch.float32)
        x = torch.randn(batch, seq, dim, dim, device='cuda', dtype=torch.float32)
        
        out_naive = ssm_scan_naive(cumsum_A, x)
        out_triton = ssm_scan_triton(cumsum_A, x)
        
        assert out_naive.shape == out_triton.shape
        
        max_diff = (out_naive - out_triton).abs().max().item()
        mean_diff = (out_naive - out_triton).abs().mean().item()
        rel_error = ((out_naive - out_triton).abs() / (out_naive.abs() + 1e-8)).max().item()
        
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        print(f"  Max relative error: {rel_error:.2e}")
        
        assert torch.allclose(out_naive, out_triton, rtol=1e-4, atol=1e-5), \
            f"Outputs don't match! Max diff: {max_diff:.2e}"
        
        print(f"  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All correctness tests PASSED!")
    print("=" * 60)


def test_performance():
    """Benchmark against the naive einsum version"""
    print("\n" + "=" * 60)
    print("Testing Performance")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    configs = [
        (2, 8, 64),      # short sequence
        (4, 64, 64),     # medium
        (2, 1024, 64),   # long - this is where we should really shine
    ]
    
    n_warmup = 5
    n_iters = 20
    
    print(f"\nWarmup: {n_warmup}, Iterations: {n_iters}")
    
    for batch, seq, dim in configs:
        print(f"\n{'='*60}")
        print(f"Config: batch={batch}, seq={seq}, dim={dim}")
        print(f"{'='*60}")
        
        cumsum_A = torch.randn(batch, seq, dim, device='cuda', dtype=torch.float32)
        x = torch.randn(batch, seq, dim, dim, device='cuda', dtype=torch.float32)
        
        # Warmup + benchmark naive
        for _ in range(n_warmup):
            _ = ssm_scan_naive(cumsum_A, x)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_iters):
            _ = ssm_scan_naive(cumsum_A, x)
        torch.cuda.synchronize()
        naive_time = (time.time() - start) / n_iters * 1000
        
        # Warmup + benchmark triton
        for _ in range(n_warmup):
            _ = ssm_scan_triton(cumsum_A, x)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_iters):
            _ = ssm_scan_triton(cumsum_A, x)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / n_iters * 1000
        
        speedup = naive_time / triton_time
        
        print(f"  Naive:  {naive_time:.3f} ms")
        print(f"  Triton: {triton_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"  ✓ Triton is {speedup:.2f}x faster!")
        else:
            print(f"  ⚠ Warning: Triton is slower")
    
    print("\n" + "=" * 60)
    print("Performance testing complete!")
    print("=" * 60)


def test_integration_with_ssm():
    """Make sure it works with the actual SSM layer"""
    print("\n" + "=" * 60)
    print("Testing Integration with SSM Layer")
    print("=" * 60)
    
    from ssm import StateSpaceLayer
    
    torch.manual_seed(42)
    dim = 16
    batch = 2
    seq = 8
    
    layer = StateSpaceLayer(dim).cuda()
    x = torch.randn(batch, seq, dim, dim, device='cuda')
    
    with torch.no_grad():
        # Replicate what happens in the forward pass
        A_continuous = -torch.exp(layer.log_A)
        dt = torch.nn.functional.softplus(layer.dt_proj(x)).squeeze(-1)
        log_A = dt * A_continuous
        cumsum_A = torch.cumsum(log_A, dim=1)
        
        # Original einsum version
        log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1)
        indices = torch.arange(seq, device=x.device)
        mask = indices[:, None] >= indices[None, :]
        log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
        M = torch.exp(log_M)
        out_original = torch.einsum('b t j r, b j r c -> b t r c', M, x)
        
        # Our kernel
        out_triton = ssm_scan_triton(cumsum_A, x)
    
    max_diff = (out_original - out_triton).abs().max().item()
    print(f"\nMax difference: {max_diff:.2e}")
    
    assert torch.allclose(out_original, out_triton, rtol=1e-4, atol=1e-5)
    
    print("✓ Integration test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_correctness()
    test_performance()
    test_integration_with_ssm()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
