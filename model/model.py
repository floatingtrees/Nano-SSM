

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1) Multi-Head SSM Mixer
#    Treat it like MultiheadAttention: (B,T,D) -> (B,T,D)
# -------------------------
class MultiHeadSSM(nn.Module):
    """
    Minimal multi-head SSM mixer.

    Input:  x (B, T, D)
    Output: y (B, T, D)

    Idea (per head, per channel):
        s_t = a_t * s_{t-1} + x_t
        where a_t = exp(dt_t * A), A < 0 for stability, dt_t > 0
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # (Optional but useful) input/output projections, like MHA has
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Stable continuous-time A: A = -exp(log_A)
        # Per-head, per-channel A
        self.log_A = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)

        # dt projection: produce one dt per head per token (then broadcast to channels)
        # dt > 0 by softplus
        self.dt_proj = nn.Linear(d_model, n_heads, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """
        x: (B, T, D)
        state (optional): (B, n_heads, head_dim) for streaming/inference
        returns:
            y: (B, T, D)
            new_state: (B, n_heads, head_dim)
        """
        B, T, D = x.shape
        x = self.in_proj(x)  # (B,T,D)

        # reshape to heads
        xh = x.view(B, T, self.n_heads, self.head_dim)  # (B,T,H,hd)

        # A_continuous: (H,hd), negative
        A_cont = -torch.exp(self.log_A)  # (H,hd)

        # dt: (B,T,H), positive
        dt = F.softplus(self.dt_proj(x))  # (B,T,H)

        # a_t = exp(dt_t * A)  -> (B,T,H,hd)
        # broadcast dt over head_dim
        a = torch.exp(dt.unsqueeze(-1) * A_cont.unsqueeze(0).unsqueeze(0))

        # init state
        if state is None:
            s = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        else:
            s = state

        ys = []
        # O(T) scan (simple + correct). Later you can optimize without changing interface.
        for t in range(T):
            s = a[:, t] * s + xh[:, t]          # (B,H,hd)
            ys.append(s)

        y = torch.stack(ys, dim=1)              # (B,T,H,hd)
        y = y.reshape(B, T, D)                  # (B,T,D)
        y = self.out_proj(self.dropout(y))      # (B,T,D)

        return y, s


# -------------------------
# 2) Standard Transformer-style Block (Pre-LN)
#    LN at start of each sublayer, no mask
# -------------------------
class SSMTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mixer = MultiHeadSSM(d_model, n_heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(d_model)
        hidden = mlp_ratio * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        # Pre-LN + SSM mixer
        y, new_state = self.mixer(self.ln1(x), state=state)
        x = x + y

        # Pre-LN + MLP
        x = x + self.mlp(self.ln2(x))
        return x, new_state


# -------------------------
# 3) Full LM: (B,T) -> (B,T,V)
# -------------------------
class SSMTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6,
                 n_heads: int = 8, max_seq_len: int = 2048,
                 mlp_ratio: int = 4, dropout: float = 0.0,
                 tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SSMTransformerBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor, states: list[torch.Tensor] | None = None):
        """
        input_ids: (B,T)
        states (optional): list of per-layer states, each (B,H,hd) for streaming/inference
        returns:
            logits: (B,T,V)
            new_states: list of states per layer
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, "Sequence length exceeds max_seq_len"

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1,T)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)              # (B,T,D)
        x = self.drop(x)

        new_states = []
        if states is None:
            states = [None] * len(self.blocks)

        for blk, st in zip(self.blocks, states):
            x, st_new = blk(x, state=st)
            new_states.append(st_new)

        x = self.ln_f(x)                     # (B,T,D)
        logits = self.lm_head(x)             # (B,T,V)
        return logits, new_states


# -------------------------
# 4) Quick shape test (what your screenshot asked for)
# -------------------------
def shape_sanity_check():
    torch.manual_seed(0)
    B, T, V = 2, 16, 1000
    model = SSMTransformerLM(vocab_size=V, d_model=128, n_layers=2, n_heads=8, max_seq_len=256)
    input_ids = torch.randint(0, V, (B, T))
    logits, _ = model(input_ids)
    assert logits.shape == (B, T, V), f"Expected {(B,T,V)}, got {logits.shape}"
    print("shape ok:", logits.shape)

if __name__ == "__main__":
    shape_sanity_check()
