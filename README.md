# nanoSSM - Optimized State Space Model Kernels

Fast Triton kernels for computing parallel SSM scans without materializing quadratic-sized intermediate tensors.

## The Problem

Given:
- `cumsum_A`: `(batch, seq, dim)` - cumulative log decay factors
- `x`: `(batch, seq, dim, dim)` - input sequence

Compute:
```
out[b,t,r,c] = sum_{j=0}^{t} exp(A[b,t,r] - A[b,j,r]) * x[b,j,r,c]
```

## Naive Implementation (ssm.py)

The straightforward PyTorch approach:
```python
log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1)  # (batch, seq, seq, dim)
M = torch.exp(log_M)  # Decay matrix
out = torch.einsum('b t j r, b j r c -> b t r c', M, x)
```

**Problem**: Creates `(batch, seq, seq, dim)` tensor → O(seq²) memory!

## Our Triton Kernel (kernel.py)

Computes outputs on-the-fly without materializing the decay matrix:
- **Memory**: O(batch * seq * dim²) instead of O(batch * seq² * dim)
- **Speed**: ~3x faster than einsum on typical workloads
- **Parallelization**: One GPU thread per (batch, timestep, row)

## Performance Results

```
Config: batch=2, seq=8, dim=64
  Naive:  0.046 ms
  Triton: 0.015 ms → 3.12x faster

Config: batch=2, seq=1024, dim=64
  Naive:  14.165 ms
  Triton: 4.617 ms → 3.07x faster
```

## Complexity: O(n²) vs O(n log n)

### Current Implementation: O(n²)
While we avoid materializing the decay matrix, each output timestep still requires summing over all previous timesteps:
```
for t in range(seq):           # O(seq)
    for j in range(t+1):       # O(seq) on average
        out[t] += decay * x[j]
```
Total: **O(seq²) per batch/row/col**

### Why Not O(n log n)?

True O(n log n) requires a **parallel associative scan** (prefix sum). For a linear recurrence like:
```
y[t] = A[t] * y[t-1] + x[t]
```

This can be computed in O(log n) depth using binary tree reduction.

**Challenges for our problem:**
1. **Operator complexity**: Need to track `(decay, weighted_sum)` pairs with associative operator `(A,x) ⊗ (B,y) = (B*A, B*x + y)`
2. **Multi-kernel coordination**: Requires up-sweep + down-sweep phases with careful synchronization
3. **4D tensor structure**: Scanning over (batch, seq, dim, dim) complicates bookkeeping

**Implementation effort**: High complexity for modest practical gain, since:
- GPU parallelism already saturated with current approach
- Memory is saved regardless of O(n²) vs O(n log n) time
- Real bottleneck is often the recurrence itself, not computation overhead

## Usage

```python
from kernel import ssm_scan_triton

# Your SSM forward pass
cumsum_A = torch.cumsum(log_A, dim=1)
out = ssm_scan_triton(cumsum_A, x)  # Fast, memory-efficient

# Or use the experimental associative scan (still O(n²) placeholder)
out = ssm_scan_triton(cumsum_A, x, use_assoc_scan=True)
```

## Files

- `kernel.py` - Optimized Triton kernels
- `ssm.py` - Full SSM layer with both naive and kernel-accelerated forward
- `tests.py` - Correctness and performance tests
- `ssm_simple.py` - Reference implementation

## Testing

```bash
python tests.py
```

Runs correctness tests (multiple shapes) and performance benchmarks.

## Future Work

For true O(n log n) complexity:
1. Implement full parallel prefix scan with up-sweep/down-sweep
2. Handle 4D tensor structure efficiently
3. Benchmark at very long sequences (seq > 10K) to see if complexity wins over constants

For now, the simple O(n²) kernel provides excellent practical performance! 🚀
