import torch
import torch.nn as nn
from math import sqrt
from einspace.layers import EinLinear, EinNorm


# === Multi-Head Self-Attention ===
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = EinLinear(embed_dim, embed_dim)
        self.k_proj = EinLinear(embed_dim, embed_dim)
        self.v_proj = EinLinear(embed_dim, embed_dim)
        self.out_proj = EinLinear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, T] → [B, T, C]
        x = x.permute(0, 2, 1)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        B, T, C = Q.shape
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_output)
        out = self.norm(x + out)  # Residual + Norm
        return out.permute(0, 2, 1)  # Back to [B, C, T]


mha_self_attention = MultiHeadSelfAttention


# === Feedforward Block ===
class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=128, **kwargs):
        super().__init__()
        self.ff = nn.Sequential(
            EinLinear(embed_dim, hidden_dim),
            nn.ReLU(),
            EinLinear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.norm(x + self.ff(x))
        return out.permute(0, 2, 1)


ff_block = FeedForwardBlock


# === Transformer Encoder Block ===
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, ff_hidden_dim=128, **kwargs):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = FeedForwardBlock(embed_dim, ff_hidden_dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x


transformer_encoder_block = TransformerEncoderBlock


# === Norm wrapper using your existing EinNorm ===
def transformer_norm(**kwargs):
    return EinNorm(**kwargs)
