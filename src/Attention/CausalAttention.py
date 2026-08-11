import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",
                             torch.tril(torch.ones(context_length, context_length),
                             diagonal=1))
        # Note: The use of `register_buffer` is important here (although it's not strictly necessary).
        # This ensures that buffers are automatically moved to the appropriate device (CPU verse GPU)

    def forward(self, x):
        # we transpose dimension 1 and 2, keeping the batch dimension at the first position
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # we transpose dimension 1 and 2, keeping the batch dimension at the first position
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_( # Trailing underscore are performed in-place avoiding memory allocations.
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ values
        return context_vector