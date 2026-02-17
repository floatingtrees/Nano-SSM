
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ssm_simple import StateSpaceLayer


# -------------------------
# 2) Standard Transformer-style Block (Pre-LN)
#    LN at start of each sublayer, no mask
# -------------------------
class SSMTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        # Replace missing MultiHeadSSM with SSMMixer that uses existing
        # StateSpaceLayer implementations (head-wise SSM applied to each head).
        self.mixer = SSMMixer(d_model, n_heads, dropout=dropout)

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


class SSMMixer(nn.Module):
    """Head-wise SSM mixer that applies a small StateSpaceLayer per head.

    This avoids multi-head attention and reuses `StateSpaceLayer` from
    `model/ssm_simple.py`. It projects features into heads, applies
    the SSM on each head treating the feature vector as shape (B,T,head_dim,1),
    then merges heads back.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for SSMMixer")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # Create one StateSpaceLayer per head (operates on head_dim)
        self.ssm_heads = nn.ModuleList([StateSpaceLayer(self.head_dim) for _ in range(n_heads)])

    def forward(self, x: torch.Tensor, state: list[torch.Tensor] | None = None):
        # x: (B,T,D)
        B, T, D = x.shape
        x = self.in_proj(x)  # (B,T,D)

        # Split into heads: (B,T,n_heads,head_dim) -> (n_heads, B, T, head_dim)
        h = x.view(B, T, self.n_heads, self.head_dim).permute(2, 0, 1, 3)

        out_heads = []
        new_states = []
        for i, head_layer in enumerate(self.ssm_heads):
            head_x = h[i]  # (B,T,head_dim)

            # StateSpaceLayer expects (B, T, dim, c). We use c=1 (feature channels)
            head_in = head_x.unsqueeze(-1)  # (B,T,head_dim,1)

            # pass through SSM head
            head_out = head_layer(head_in)  # (B,T,head_dim,1)

            # reduce last dim
            head_out = head_out.squeeze(-1)  # (B,T,head_dim)

            out_heads.append(head_out)
            new_states.append(None)

        # Stack heads back: (n_heads, B, T, head_dim) -> (B, T, D)
        out = torch.cat([h for h in out_heads], dim=-1)
        out = self.dropout(out)
        out = self.out_proj(out)

        return out, new_states


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
