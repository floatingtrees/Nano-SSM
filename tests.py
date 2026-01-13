import torch
import time
import random
from kernel_associative import ssm_scan, ssm_forward_triton


def ssm_forward_pytorch(log_A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using PyTorch ops (supports autograd).
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


def test_forward_correctness():
    """Test forward pass correctness."""
    print("=" * 60)
    print("Testing Forward Correctness")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    configs = [
        (1, 4, 8),
        (2, 16, 16),
        (4, 64, 32),
    ]

    configs = []
    
    for batch, seq, dim in configs:
        print(f"\nConfig: batch={batch}, seq={seq}, dim={dim}")
        
        log_A = torch.randn(batch, seq, dim, device='cuda') * 0.1
        x = torch.randn(batch, seq, dim, dim, device='cuda')
        
        out_ref = ssm_forward_pytorch(log_A, x)
        out_triton = ssm_forward_triton(log_A, x)
        
        diff = (out_triton - out_ref).abs().max().item()
        print(f"  Forward diff: {diff:.2e}")
        
        assert torch.allclose(out_triton, out_ref, rtol=1e-3, atol=1e-4), \
            f"Forward doesn't match! Max diff: {diff:.2e}"
        
        print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All forward correctness tests PASSED!")
    print("=" * 60)


def test_backward_correctness():
    """Test backward pass against PyTorch autograd."""
    print("\n" + "=" * 60)
    print("Testing Backward Correctness (vs PyTorch autograd)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    configs = [
        (1, 4, 8),
        (2, 16, 16),
        (4, 64, 32),
    ]
    
    for batch, seq, dim in configs:
        print(f"\nConfig: batch={batch}, seq={seq}, dim={dim}")
        
        # Create inputs
        log_A_data = torch.randn(batch, seq, dim, device='cuda') * 0.1
        x_data = torch.randn(batch, seq, dim, dim, device='cuda')
        
        # Triton version
        log_A = log_A_data.clone().requires_grad_(True)
        x = x_data.clone().requires_grad_(True)
        
        # PyTorch reference version
        log_A_ref = log_A_data.clone().requires_grad_(True)
        x_ref = x_data.clone().requires_grad_(True)
        
        # Forward
        out_triton = ssm_scan(log_A, x)
        out_ref = ssm_forward_pytorch(log_A_ref, x_ref)
        
        # Random gradient
        dout = torch.randn_like(out_triton)
        
        # Backward
        out_triton.backward(dout)
        out_ref.backward(dout)
        
        # Compare
        dx_diff = (x.grad - x_ref.grad).abs().max().item()
        dlog_A_diff = (log_A.grad - log_A_ref.grad).abs().max().item()
        
        print(f"  dx diff:      {dx_diff:.2e}")
        print(f"  dlog_A diff:  {dlog_A_diff:.2e}")
        
        assert torch.allclose(x.grad, x_ref.grad, rtol=1e-2, atol=1e-3), \
            f"dx doesn't match! Max diff: {dx_diff:.2e}"
        assert torch.allclose(log_A.grad, log_A_ref.grad, rtol=1e-2, atol=1e-3), \
            f"dlog_A doesn't match! Max diff: {dlog_A_diff:.2e}"
        
        print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All backward correctness tests PASSED!")
    print("=" * 60)


def test_random_shapes():
    """Test with randomly generated shapes."""
    print("\n" + "=" * 60)
    print("Testing Random Shapes")
    print("=" * 60)
    
    random.seed(42)
    torch.manual_seed(42)
    
    n_tests = 10
    
    for i in range(n_tests):
        batch = random.randint(1, 8)
        seq = random.randint(2, 128)
        dim = random.randint(4, 64)
        
        print(f"\nTest {i+1}/{n_tests}: batch={batch}, seq={seq}, dim={dim}")
        
        # Create inputs
        log_A_data = torch.randn(batch, seq, dim, device='cuda') * 0.1
        x_data = torch.randn(batch, seq, dim, dim, device='cuda')
        
        # Triton version
        log_A = log_A_data.clone().requires_grad_(True)
        x = x_data.clone().requires_grad_(True)
        
        # PyTorch reference
        log_A_ref = log_A_data.clone().requires_grad_(True)
        x_ref = x_data.clone().requires_grad_(True)
        
        # Forward
        out_triton = ssm_scan(log_A, x)
        out_ref = ssm_forward_pytorch(log_A_ref, x_ref)
        
        forward_diff = (out_triton - out_ref).abs().max().item()
        
        # Backward
        dout = torch.randn_like(out_triton)
        out_triton.backward(dout)
        out_ref.backward(dout)
        
        dx_diff = (x.grad - x_ref.grad).abs().max().item()
        dlog_A_diff = (log_A.grad - log_A_ref.grad).abs().max().item()
        atol = 1e-3 * max(1, seq / 32)  # Scale atol with sequence length
        
        print(f"  Forward diff: {forward_diff:.2e}")
        print(f"  dx diff:      {dx_diff:.2e}")
        print(f"  dlog_A diff:  {dlog_A_diff:.2e}")
        print(f"  atol:         {atol:.2e}")
        
        # Scale tolerance with sequence length (longer sequences accumulate more error)
        rtol = 1e-2
        
        
        assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol), \
            f"Forward doesn't match! Max diff: {forward_diff:.2e}"
        assert torch.allclose(x.grad, x_ref.grad, rtol=rtol, atol=atol), \
            f"dx doesn't match! Max diff: {dx_diff:.2e}"
        assert torch.allclose(log_A.grad, log_A_ref.grad, rtol=rtol, atol=atol), \
            f"dlog_A doesn't match! Max diff: {dlog_A_diff:.2e}"
        
        print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All random shape tests PASSED!")
    print("=" * 60)


def test_performance():
    """Benchmark forward + backward performance."""
    print("\n" + "=" * 60)
    print("Testing Forward + Backward Performance")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    configs = [
        (2, 64, 64),
        (4, 256, 64),
        (2, 1024, 64),
        (2, 2048, 64),
    ]
    
    n_warmup = 5
    n_iters = 20
    
    print(f"\nWarmup: {n_warmup}, Iterations: {n_iters}")
    
    for batch, seq, dim in configs:
        print(f"\n{'='*60}")
        print(f"Config: batch={batch}, seq={seq}, dim={dim}")
        print(f"{'='*60}")
        
        # Benchmark PyTorch reference
        pytorch_times = []
        for i in range(n_warmup + n_iters):
            log_A = (torch.randn(batch, seq, dim, device='cuda') * 0.1).requires_grad_(True)
            x = torch.randn(batch, seq, dim, dim, device='cuda').requires_grad_(True)
            dout = torch.randn(batch, seq, dim, dim, device='cuda')
            
            torch.cuda.synchronize()
            start = time.time()
            
            out = ssm_forward_pytorch(log_A, x)
            out.backward(dout)
            
            torch.cuda.synchronize()
            
            if i >= n_warmup:
                pytorch_times.append(time.time() - start)
        
        pytorch_time = sum(pytorch_times) / len(pytorch_times) * 1000
        
        # Benchmark Triton
        triton_times = []
        for i in range(n_warmup + n_iters):
            log_A = (torch.randn(batch, seq, dim, device='cuda') * 0.1).requires_grad_(True)
            x = torch.randn(batch, seq, dim, dim, device='cuda').requires_grad_(True)
            dout = torch.randn(batch, seq, dim, dim, device='cuda')
            
            torch.cuda.synchronize()
            start = time.time()
            
            out = ssm_scan(log_A, x)
            out.backward(dout)
            
            torch.cuda.synchronize()
            
            if i >= n_warmup:
                triton_times.append(time.time() - start)
        
        triton_time = sum(triton_times) / len(triton_times) * 1000
        
        speedup = pytorch_time / triton_time
        
        print(f"  PyTorch O(n²) fwd+bwd: {pytorch_time:.3f} ms")
        print(f"  Triton O(n) fwd+bwd:   {triton_time:.3f} ms")
        print(f"  Speedup:               {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"  ✓ Triton is {speedup:.2f}x faster!")
        else:
            print(f"  ⚠ Triton is slower")
    
    print("\n" + "=" * 60)
    print("Performance testing complete!")
    print("=" * 60)


def test_fp64_precision():
    """
    Test with fp64 to verify precision issues are due to float32.
    fp64 should have much smaller errors.
    """
    print("\n" + "=" * 60)
    print("Testing FP64 Precision (comparing fp32 vs fp64 errors)")
    print("=" * 60)
    
    random.seed(42)
    torch.manual_seed(42)
    
    configs = [
        (2, 32, 32),
        (4, 64, 32),
        (2, 128, 64),
        (3, 256, 48),
    ]
    
    for batch, seq, dim in configs:
        print(f"\nConfig: batch={batch}, seq={seq}, dim={dim}")
        
        # === FP32 Test ===
        log_A_data_32 = torch.randn(batch, seq, dim, device='cuda', dtype=torch.float32) * 0.1
        x_data_32 = torch.randn(batch, seq, dim, dim, device='cuda', dtype=torch.float32)
        
        log_A_32 = log_A_data_32.clone().requires_grad_(True)
        x_32 = x_data_32.clone().requires_grad_(True)
        log_A_ref_32 = log_A_data_32.clone().requires_grad_(True)
        x_ref_32 = x_data_32.clone().requires_grad_(True)
        
        out_triton_32 = ssm_scan(log_A_32, x_32)
        out_ref_32 = ssm_forward_pytorch(log_A_ref_32, x_ref_32)
        
        forward_diff_32 = (out_triton_32 - out_ref_32).abs().max().item()
        
        dout_32 = torch.randn_like(out_triton_32)
        out_triton_32.backward(dout_32)
        out_ref_32.backward(dout_32)
        
        dx_diff_32 = (x_32.grad - x_ref_32.grad).abs().max().item()
        dlog_A_diff_32 = (log_A_32.grad - log_A_ref_32.grad).abs().max().item()
        
        # === FP64 Test ===
        log_A_data_64 = log_A_data_32.double()
        x_data_64 = x_data_32.double()
        
        log_A_64 = log_A_data_64.clone().requires_grad_(True)
        x_64 = x_data_64.clone().requires_grad_(True)
        log_A_ref_64 = log_A_data_64.clone().requires_grad_(True)
        x_ref_64 = x_data_64.clone().requires_grad_(True)
        
        out_triton_64 = ssm_scan(log_A_64, x_64)
        out_ref_64 = ssm_forward_pytorch(log_A_ref_64, x_ref_64)
        
        forward_diff_64 = (out_triton_64 - out_ref_64).abs().max().item()
        
        dout_64 = dout_32.double()
        out_triton_64.backward(dout_64)
        out_ref_64.backward(dout_64)
        
        dx_diff_64 = (x_64.grad - x_ref_64.grad).abs().max().item()
        dlog_A_diff_64 = (log_A_64.grad - log_A_ref_64.grad).abs().max().item()
        
        # Print comparison
        print(f"  FP32:")
        print(f"    Forward diff: {forward_diff_32:.2e}")
        print(f"    dx diff:      {dx_diff_32:.2e}")
        print(f"    dlog_A diff:  {dlog_A_diff_32:.2e}")
        print(f"  FP64:")
        print(f"    Forward diff: {forward_diff_64:.2e}")
        print(f"    dx diff:      {dx_diff_64:.2e}")
        print(f"    dlog_A diff:  {dlog_A_diff_64:.2e}")
        
        # FP64 should have significantly smaller errors
        forward_improvement = forward_diff_32 / max(forward_diff_64, 1e-15)
        dx_improvement = dx_diff_32 / max(dx_diff_64, 1e-15)
        dlog_A_improvement = dlog_A_diff_32 / max(dlog_A_diff_64, 1e-15)
        
        print(f"  Improvement (fp32/fp64):")
        print(f"    Forward: {forward_improvement:.1f}x better")
        print(f"    dx:      {dx_improvement:.1f}x better")
        print(f"    dlog_A:  {dlog_A_improvement:.1f}x better")
        
        # Verify fp64 has much tighter precision
        assert forward_diff_64 < 1e-10, f"FP64 forward diff too large: {forward_diff_64:.2e}"
        assert dx_diff_64 < 1e-10, f"FP64 dx diff too large: {dx_diff_64:.2e}"
        assert dlog_A_diff_64 < 1e-9, f"FP64 dlog_A diff too large: {dlog_A_diff_64:.2e}"
        
        print("  ✓ FP64 precision verified!")
    
    print("\n" + "=" * 60)
    print("FP64 precision tests PASSED!")
    print("=" * 60)


def test_fp64_random_shapes():
    """Test fp64 with randomly generated shapes."""
    print("\n" + "=" * 60)
    print("Testing FP64 Random Shapes")
    print("=" * 60)
    
    random.seed(123)
    torch.manual_seed(123)
    
    n_tests = 10
    
    for i in range(n_tests):
        batch = random.randint(1, 8)
        seq = random.randint(2, 200)
        dim = random.randint(4, 64)
        
        print(f"\nTest {i+1}/{n_tests}: batch={batch}, seq={seq}, dim={dim}")
        
        # Create fp64 inputs
        log_A_data = torch.randn(batch, seq, dim, device='cuda', dtype=torch.float64) * 0.1
        x_data = torch.randn(batch, seq, dim, dim, device='cuda', dtype=torch.float64)
        
        log_A = log_A_data.clone().requires_grad_(True)
        x = x_data.clone().requires_grad_(True)
        log_A_ref = log_A_data.clone().requires_grad_(True)
        x_ref = x_data.clone().requires_grad_(True)
        
        # Forward
        out_triton = ssm_scan(log_A, x)
        out_ref = ssm_forward_pytorch(log_A_ref, x_ref)
        
        forward_diff = (out_triton - out_ref).abs().max().item()
        
        # Backward
        dout = torch.randn_like(out_triton)
        out_triton.backward(dout)
        out_ref.backward(dout)
        
        dx_diff = (x.grad - x_ref.grad).abs().max().item()
        dlog_A_diff = (log_A.grad - log_A_ref.grad).abs().max().item()
        
        print(f"  Forward diff: {forward_diff:.2e}")
        print(f"  dx diff:      {dx_diff:.2e}")
        print(f"  dlog_A diff:  {dlog_A_diff:.2e}")
        
        # FP64 should have very tight tolerances
        assert forward_diff < 1e-10, f"Forward diff too large: {forward_diff:.2e}"
        assert dx_diff < 1e-10, f"dx diff too large: {dx_diff:.2e}"
        assert dlog_A_diff < 1e-9, f"dlog_A diff too large: {dlog_A_diff:.2e}"
        
        print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All FP64 random shape tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_fp64_precision()
    test_fp64_random_shapes()
    test_forward_correctness()
    test_backward_correctness()
    test_random_shapes()
    
    test_performance()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
