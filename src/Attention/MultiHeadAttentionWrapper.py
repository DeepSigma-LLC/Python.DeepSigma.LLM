import torch
import torch.nn as nn

from src.Attention.CausalAttention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    """
    Simple wrapper around multiple causal attention layers.
    Simply initializes multiple causal attention layers and concatenates their outputs.
    """
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)