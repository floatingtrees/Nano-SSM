import torch
import torch.nn as nn
import torch.nn.functional as F

class StateSpaceLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_A = nn.Parameter((torch.randn(dim) * 0.02))
        self.dt_proj = nn.Linear(dim, 1) 
        
        self.norm = nn.LayerNorm((dim, dim), eps=1e-5)

    def recurrent_form(self, x, state):
        x = self.norm(x)
        A_continuous = -torch.exp(self.log_A)
        dt = F.softplus(self.dt_proj(x)).squeeze(-1) 
        A_discrete = torch.exp(dt * A_continuous)
        return A_discrete.unsqueeze(-1) * state + x
    
    def forward(self, x):
        """
        Parallel Forward Pass (Convolution/Scan style).
        Args: x: (b, seq, d, d)
        Returns: out: (b, seq, d, d)
        """
        x_norm = self.norm(x) 

        b, seq, d, _ = x_norm.shape
        A_continuous = -torch.exp(self.log_A)
        dt = F.softplus(self.dt_proj(x_norm)).squeeze(-1)
        log_A = dt * A_continuous 
        
        cumsum_A = torch.cumsum(log_A, dim=1)
        
        log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1) 
        
        indices = torch.arange(seq, device=x.device)
        mask = indices[:, None] >= indices[None, :]
        log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
        
        M = torch.exp(log_M)

        out = torch.einsum('b t j r, b j r c -> b t r c', M, x_norm)
        
        return out


class MatrixFeedForward(nn.Module):
    """
    Mixes information across the dimensions of the matrix state.
    Since the SSM mixes time, this mixes the 'features' (d x d).
    """
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        input_dim = dim * dim
        hidden_dim = input_dim * expansion_factor
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm((dim, dim))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        b, s, d, _ = x.shape
        x = x.view(b, s, -1)
        
        x = self.net(x)
        
        x = x.view(b, s, d, d)
        return residual + x

# --- 3. The Residual Block ---
class MatrixSSMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ssm = StateSpaceLayer(dim)
        self.ffn = MatrixFeedForward(dim)
        
      
    def forward(self, x):
 
        ssm_out = self.ssm(x) 
        x = x + ssm_out
        

        x = self.ffn(x) 
        
        return x

    def step(self, x, state):
        """Step function for autoregressive generation"""

        next_state = self.ssm.recurrent_form(x, state)
        

        post_ssm = x + next_state
        
        # FFN
        out = self.ffn(post_ssm.unsqueeze(1)).squeeze(1)
        
        return out, next_state

# --- 4. The Main Model ---
class MatrixSSMModel(nn.Module):
    def __init__(self, vocab_size, dim, depth=4):
        """
        Args:
            vocab_size: Size of vocabulary.
            dim: The dimension 'd' of the square matrix state.
                 Effective embedding size will be d*d.
            depth: Number of stacked blocks.
        """
        super().__init__()
        self.dim = dim
        self.embedding_dim = dim * dim
        
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        
        self.layers = nn.ModuleList([
            MatrixSSMBlock(dim) for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(self.embedding_dim)
        self.classifier = nn.Linear(self.embedding_dim, vocab_size, bias=False)

    def forward(self, input_ids):

        
        x = self.embedding(input_ids) 
        b, s, _ = x.shape
        x = x.view(b, s, self.dim, self.dim)
        

        for layer in self.layers:
            x = layer(x)
            

        x = x.view(b, s, -1)
        x = self.final_norm(x)
        logits = self.classifier(x)
        
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20):
        """
        Autoregressive generation using the recurrent form.
        demonstrates the O(1) inference capability.
        """
        self.eval()
        
    
        states = [torch.zeros(input_ids.shape[0], self.dim, self.dim, device=input_ids.device) 
                  for _ in self.layers]
        

        
        current_input_ids = input_ids
        

        x = self.embedding(current_input_ids)
        x_matrix = x.view(x.shape[0], x.shape[1], self.dim, self.dim)
        

        for t in range(x_matrix.shape[1]):
            step_input = x_matrix[:, t, :, :]
            for i, layer in enumerate(self.layers):
                step_input, states[i] = layer.step(step_input, states[i])

        next_token_input = step_input
        
        generated = []
        
        for _ in range(max_new_tokens):

            next_token_flat = next_token_input.view(input_ids.shape[0], -1)
            logits = self.classifier(self.final_norm(next_token_flat))
            

            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token_id)
            
            x_next = self.embedding(next_token_id).squeeze(1) # (b, d*d)
            step_input = x_next.view(input_ids.shape[0], self.dim, self.dim)
            

            for i, layer in enumerate(self.layers):
                step_input, states[i] = layer.step(step_input, states[i])
            
            next_token_input = step_input

        return torch.cat(generated, dim=1)


def test_equivalence():

    b, seq, d = 2, 5, 4
    vocab_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MatrixSSMModel(vocab_size=vocab_size, dim=d).to(device)
    model.eval()


    input_ids = torch.randint(0, vocab_size, (b, seq)).to(device)

    with torch.no_grad():
        forward_logits = model(input_ids) 
    

    recurrent_outputs = []
    

    states = [torch.zeros(b, d, d, device=device) for _ in model.layers]
    

    x = model.embedding(input_ids)
    x_matrix = x.view(b, seq, d, d)

    with torch.no_grad():
        for t in range(seq):
            step_input = x_matrix[:, t, :, :]
            
            # Pass through each block using the .step() methd
            for i, layer_block in enumerate(model.layers):
                step_input, states[i] = layer_block.step(step_input, states[i])
            

            final_step_vec = step_input.view(b, -1)
            logits_step = model.classifier(model.final_norm(final_step_vec))
            recurrent_outputs.append(logits_step.unsqueeze(1))

    recurrent_logits = torch.cat(recurrent_outputs, dim=1)

    print(f"Forward Shape:   {forward_logits.shape}")
    print(f"Recurrent Shape: {recurrent_logits.shape}")

    assert forward_logits.shape == recurrent_logits.shape, " Shapes do not match!"


    is_close = torch.allclose(forward_logits, recurrent_logits, atol=1e-4)
    
    if is_close:
        print("works.")
    else:
        diff = (forward_logits - recurrent_logits).abs().max()
        print(f"not the same: {diff.item()}")

test_equivalence()
