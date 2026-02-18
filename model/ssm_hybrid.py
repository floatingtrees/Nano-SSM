import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. The Cache Container ---
class InferenceCache:
    """
    Holds the state for SSMs and the Key/Values for Attention
    to speed up autoregressive generation.
    """
    def __init__(self, batch_size, dim, device):
        self.ssm_states = {}  # Map layer_idx -> Tensor (b, d, d)
        self.attn_kv = {}     # Map layer_idx -> (K, V)
        
        self.batch_size = batch_size
        self.dim = dim
        self.device = device

    def get_ssm_state(self, layer_idx):
        if layer_idx not in self.ssm_states:
            # Initialize SSM state as zeros (b, d, d)
            self.ssm_states[layer_idx] = torch.zeros(
                self.batch_size, self.dim, self.dim, device=self.device
            )
        return self.ssm_states[layer_idx]

    def update_ssm_state(self, layer_idx, new_state):
        self.ssm_states[layer_idx] = new_state

    def get_attn_kv(self, layer_idx):
        return self.attn_kv.get(layer_idx, (None, None))

    def update_attn_kv(self, layer_idx, k, v):
        self.attn_kv[layer_idx] = (k, v)


# --- 2. Core Layers ---

class StateSpaceLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Initialize A with small random values
        self.log_A = nn.Parameter((torch.randn(dim) * 0.02))
        self.dt_proj = nn.Linear(dim, 1) 
        self.norm = nn.LayerNorm((dim, dim), eps=1e-5)
        
    def forward(self, x):
        """
        Parallel Forward Pass (Convolution/Scan style) for Training.
        Input: (b, seq, d, d)
        """
        x_in = x
        x = self.norm(x)
        
        b, seq, d, _ = x.shape
        A_continuous = -torch.exp(self.log_A)
        
        dt = F.softplus(self.dt_proj(x)).squeeze(-1) 
        log_A = dt * A_continuous 
        
        cumsum_A = torch.cumsum(log_A, dim=1)
        log_M = cumsum_A.unsqueeze(2) - cumsum_A.unsqueeze(1) 
        
        indices = torch.arange(seq, device=x.device)
        mask = indices[:, None] >= indices[None, :]
        log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
        
        M = torch.exp(log_M)

        out = torch.einsum('b t j r, b j r c -> b t r c', M, x)
        
        return out + x_in

    def step(self, x, state):
        """
        Recurrent Step for Inference.
        Replaces 'recurrent_form' to correctly handle residuals.
        
        Args:
            x: (b, d, d) - Single time step input
            state: (b, d, d) - Previous hidden state
        Returns:
            layer_out: (b, d, d) - The output to pass to next layer
            next_state: (b, d, d) - The updated state to save to cache
        """
        x_in = x
        x_norm = self.norm(x)

        # Discretize A for this specific step
        A_continuous = -torch.exp(self.log_A)
        dt = F.softplus(self.dt_proj(x_norm)).squeeze(-1) 
        A_discrete = torch.exp(dt * A_continuous) # (b, d)
        
        # SSM Update Rule: h_t = A_bar * h_{t-1} + x_t
        next_state = A_discrete.unsqueeze(-1) * state + x_norm
        
        # Layer Output = SSM_Output + Residual
        layer_out = next_state + x_in
        
        return layer_out, next_state


class MatrixAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.embed_dim = dim * dim
        
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.norm = nn.LayerNorm((dim, dim))

    def forward(self, x, cache=None, layer_idx=None):
        b, seq, d, _ = x.shape
        residual = x
        
        x = self.norm(x)
        x_flat = x.view(b, seq, -1) 
        
        qkv = self.qkv_proj(x_flat).reshape(b, seq, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # --- KV Cache Logic ---
        if cache is not None:
            past_k, past_v = cache.get_attn_kv(layer_idx)
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            cache.update_attn_kv(layer_idx, k, v)
        # ----------------------

        # Causal Attention
        is_causal = True if cache is None else False 
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        attn_out = attn_out.transpose(1, 2).reshape(b, seq, -1)
        attn_out = self.out_proj(attn_out)
        
        return residual + attn_out.view(b, seq, d, d)


class MatrixFeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        input_dim = dim * dim
        hidden_dim = input_dim * expansion_factor
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
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




class HybridBlock(nn.Module):
    def __init__(self, dim, layer_type='ssm', num_heads=4):
        super().__init__()
        self.layer_type = layer_type
        
        if layer_type == 'ssm':
            self.mixer = StateSpaceLayer(dim)
        elif layer_type == 'attn':
            self.mixer = MatrixAttentionBlock(dim, num_heads=num_heads)
        
        self.ffn = MatrixFeedForward(dim)

    def forward(self, x, cache=None, layer_idx=None):
        # 1. Mixer Step (SSM or Attention)
        if self.layer_type == 'ssm':
            if cache is not None:
                # Check if this is Generation (1 token) or Prefill (Multiple tokens)
                if x.shape[1] == 1:
                    # --- FAST PATH (Generation) ---
                    # Process just one step
                    x_step = x.squeeze(1) 
                    state = cache.get_ssm_state(layer_idx)
                    out, next_state = self.mixer.step(x_step, state)
                    cache.update_ssm_state(layer_idx, next_state)
                    x = out.unsqueeze(1) 
                else:
                    # --- PREFILL PATH (Prompt Processing) ---
                    # We must loop sequentially to calculate the correct state
                    # for the cache, so the model has "memory" of the prompt.
                    b, seq, d, _ = x.shape
                    out_list = []
                    state = cache.get_ssm_state(layer_idx)
                    
                    for t in range(seq):
                        x_step = x[:, t, :, :]
                        out_step, state = self.mixer.step(x_step, state)
                        out_list.append(out_step)
                    
                    # Save final state so generation can continue from here
                    cache.update_ssm_state(layer_idx, state)
                    x = torch.stack(out_list, dim=1) 

            else:
                # No cache (Training mode), use fast Parallel Scan
                x = self.mixer(x)
                
        elif self.layer_type == 'attn':
            # Attention handles prefill/gen naturally via the KV cache logic
            x = self.mixer(x, cache=cache, layer_idx=layer_idx)
            
        # 2. Feed Forward Step
        x = self.ffn(x)
        return x


# --- 4. The Full Language Model ---

class HybridLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers=6, attn_every=2, num_heads=4):
        super().__init__()
        self.dim = dim
        self.embed_dim = dim * dim
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_type = 'attn' if (i + 1) % attn_every == 0 else 'ssm'
            self.layers.append(HybridBlock(dim, layer_type=layer_type, num_heads=num_heads))
            
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, vocab_size, bias=False)

    def forward(self, input_ids, cache=None):
        # input_ids: (b, seq)
        x = self.embedding(input_ids)
        b, s, _ = x.shape
        x = x.view(b, s, self.dim, self.dim)
        
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache, layer_idx=i)
            
        x = x.view(b, s, -1)
        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20, temperature=1.0):
        """
        Fast Generation with KV Caching and Recurrent SSM steps.
        """
        self.eval()
        b = input_ids.shape[0]
        cache = InferenceCache(b, self.dim, input_ids.device)
        
        # 1. Prefill
        # Run prompt normally so cache picks up state at the end
        logits = self(input_ids, cache=cache)
        next_token_logits = logits[:, -1, :] / temperature
        
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = [next_token]
        
        # 2. Generation Loop
        for _ in range(max_new_tokens - 1):
            # Pass only the single new token
            logits = self(next_token, cache=cache)
            
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token)
            
        return torch.cat([input_ids] + generated, dim=1)

# --- 5. Test Block ---
if __name__ == "__main__":
    torch.manual_seed(42)
    VOCAB = 1000
    DIM = 8
    model = HybridLanguageModel(VOCAB, DIM, num_layers=4, attn_every=2)
    
    print("Running test generation...")
    input_ids = torch.randint(0, VOCAB, (1, 10))
    out = model.generate(input_ids, max_new_tokens=10)
    print(f"Output shape: {out.shape}")
    print("Success.")
