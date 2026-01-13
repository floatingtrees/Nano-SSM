import torch
from torch import nn
from einops import repeat

class StateSpaceLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Initialize A. 
        self.A = nn.Parameter((torch.randn(dim) * 0.02).clamp(min=1e-6))
        
    def recurrent_form(self, x, state):
        return self.A[:, None] * state + x
    
    def forward(self, x):
        """
        Parallel Forward Pass (Convolution/Scan style).
        
        Args:
            x: (b, seq, d, d) - Sequence of Input Matrices
            A: (b, seq, d)    - Sequence of Diagonal Decay Rates
            
        Returns:
            out: (b, seq, d, d) - Sequence of State Matrices
        """
        b, seq, d, _ = x.shape
        A = repeat(self.A, "dim -> b seq dim", b = b, seq = seq )
        # 1. Compute Cumulative Decay in Log Space
        # We want the product A_t * A_{t-1} ... * A_{j+1}
        # In log space, this is cumsum(t) - cumsum(j)
        
        # Clamp A to avoid log(0)
        log_A = torch.log(A.clamp(min=1e-6))
        
        # Cumulative sum along time (dim=1)
        # cs_A shape: (b, seq, d)
        cs_A = torch.cumsum(log_A, dim=1)
        
        # 2. Construct the Decay Matrix M
        # We need M[t, j] = Decay factor from j to t
        # Shape broadcast: (b, t, 1, d) - (b, 1, j, d) -> (b, t, j, d)
        log_M = cs_A.unsqueeze(2) - cs_A.unsqueeze(1)
        
        # 3. Apply Causal Mask
        # We only want t >= j. 
        indices = torch.arange(seq, device=x.device)
        # Mask shape: (seq, seq)
        mask = indices[:, None] >= indices[None, :]
        
        # Apply mask to log_M. (b, t, j, d)
        # We use -inf so exp() becomes 0
        log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
        
        # 4. Exponentiate to get linear decay factors
        # M[b, t, j, r] is the scalar weight for row 'r' from time 'j' to 't'
        M = torch.exp(log_M)
        
        # 5. Contract (Matrix Multiplication via Einsum)
        # Equation: H[t] = sum_{j} ( M[t,j] * X[j] )
        #
        # Dimensions:
        # M: b, t, j, r  (Batch, TimeOut, TimeIn, Rows)
        # x: b, j, r, c  (Batch, TimeIn, Rows, Cols)
        #
        # Operations:
        # - Match 'j' (Sum over input time)
        # - Match 'r' (Element-wise multiplication for rows)
        # - 'c' is independent (broadcasted)
        
        out = torch.einsum('b t j r, b j r c -> b t r c', M, x)
        
        return out

def inference(model, x):
    b, seq, dim1, dim2 = x.shape
    state = torch.zeros((b, dim1, dim2), device=x.device)
    all_states = torch.zeros(x.shape)
    for i in range(seq):
        state = model.recurrent_form(x[:, i, :, :], state)
        all_states[:, i, :, :] = state
    return state, all_states

if __name__ == "__main__":
    torch.manual_seed(42)
    dim = 16
    layer = StateSpaceLayer(dim)
    
    # Batch=2, Seq=16, Dim=8x8
    x = torch.randn(2, 8, dim, dim)
    
    # 1. Sequential Inference (Last State)
    last_state_rnn, all_states_rnn = inference(layer, x)
    
    # 2. Parallel Forward (Full Sequence)
    full_seq_parallel = layer.forward(x)
    
    # Compare
    last_state_parallel = full_seq_parallel[:, -1, :, :]
    
    print("RNN Shape:     ", last_state_rnn.shape)
    print("Parallel Shape:", last_state_parallel.shape)
    
    diff = (last_state_rnn - last_state_parallel).abs().max()
    print(f"Max Final State Difference: {diff.item():.2e}")
    diff2 = (all_states_rnn - full_seq_parallel).abs().max()
    print(f"Max All State Difference: {diff2.item():.2e}")