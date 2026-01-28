import torch
from torch import nn
import torch.nn.functional as F

class StateSpaceLayer(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()

        self.log_A = nn.Parameter((torch.randn(dim) * 0.02))
        self.dt_proj = nn.Linear(dim, 1) 

        self.norm_forward = nn.LayerNorm((dim, dim))
        self.flat_dim = dim * dim
        self.norm = nn.LayerNorm(self.flat_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.flat_dim, 
            num_heads=n_heads, 
            batch_first=True
        )
        
        # Linear to mix the result back
        self.out_proj = nn.Linear(self.flat_dim, self.flat_dim)


    def MHA_layer(self, x):
        b, seq, d, _ = x.shape
        x_flat = x.view(b, seq, -1)
        x_norm = self.norm(x_flat)
        attn_mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        return self.out_proj(attn_out).view(b, seq, d, d)


    def recurrent_form(self, x, state):
        x = self.norm_forward(x)

        A_continuous = -torch.exp(self.log_A)
        dt = F.softplus(self.dt_proj(x)).squeeze(-1)
        A_discrete = torch.exp(dt * A_continuous)
        
        return A_discrete.unsqueeze(-1) * state + x
    
    def forward(self, x):

        x = self.norm_forward(x)

        b, seq, d, _ = x.shape
        

        A_continuous = -torch.exp(self.log_A)
        dt = F.softplus(self.dt_proj(x)).squeeze(-1)
        log_A = dt * A_continuous 
        
        cumsum_A = torch.cumsum(log_A, dim=1)
        
        log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1) 
        indices = torch.arange(seq, device=x.device)
        mask_ssm = indices[:, None] >= indices[None, :]
        log_M = log_M.masked_fill(~mask_ssm.unsqueeze(0).unsqueeze(-1), float('-inf'))
        M = torch.exp(log_M)
        
        out_ssm = torch.einsum('b t j r, b j r c -> b t r c', M, x)

       
        return out_ssm



def inference(model, x):
    # x shape: (b, seq, d, d)
    b, seq, d, _ = x.shape
    
    # Initialize state with Batch size as first dim
    state = torch.zeros((b, d, d), device=x.device)
    
    # Store states list
    all_states_list = []
    
    for i in range(seq):
        # Update state using the current step's input
        state = model.recurrent_form(x[:, i, :, :], state)
        all_states_list.append(state)
        
    # Stack along dimension 1 (Sequence) to get (b, seq, d, d)
    all_states = torch.stack(all_states_list, dim=1)
    
    # Return the LAST state and the FULL sequence
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


