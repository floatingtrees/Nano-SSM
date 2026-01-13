import torch
from torch import nn
from einops import repeat
import torch.nn.functional as F

class StateSpaceLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Initialize A. 
        self.log_A = nn.Parameter((torch.randn(dim) * 0.02))
        self.dt_proj = nn.Linear(dim, 1) # Last dim is 1 because diagonal
        
    def recurrent_form(self, x, state):
        A_continuous = -torch.exp(self.log_A)
        dt = F.softplus(self.dt_proj(x)).squeeze(-1)
        A_discrete = torch.exp(dt * A_continuous)
        return A_discrete.unsqueeze(-1) * state + x
    
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
        A_continuous = -torch.exp(self.log_A)
        dt = F.softplus(self.dt_proj(x)).squeeze(-1)
        log_A = dt * A_continuous # still represents only matrix diagonals

        
        cumsum_A = torch.cumsum(log_A, dim=1)
        # Unoptimized version materializes a matrix of length seq^2, we want kernel to avoid that
        log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1) # Broadcast to expand 
        
        indices = torch.arange(seq, device=x.device)

        mask = indices[:, None] >= indices[None, :]

        log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
        
        M = torch.exp(log_M)
        

        # Reduce over the 0, A, A^2, A^3, ..., A^n dimension
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
    x = torch.ones(2, 4, dim, dim)
    
    # 1. Sequential Inference (Last State)
    last_state_rnn, all_states_rnn = inference(layer, x)

    # 2. Parallel Forward (Full Sequence)
    full_seq_parallel = layer.forward(x)
    
    # Compare
    last_state_parallel = full_seq_parallel[:, -1, :, :]
    
    print("RNN Shape:", last_state_rnn.shape)
    print("Parallel Shape:", last_state_parallel.shape)
    
    diff = (last_state_rnn - last_state_parallel).abs().max()
    print(f"Max Final State Difference: {diff.item():.2e}")
    diff2 = (all_states_rnn - full_seq_parallel).abs().max()
    print(f"Max All State Difference: {diff2.item():.2e}")