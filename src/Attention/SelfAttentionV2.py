import torch
import torch.nn as nn


class SelfAttentionV2(nn.Module):
    """
    Simple implementation of the self-attention mechanism.
    Allows for trainable attention weights.
    Linear layers use a more sophisticated weight initialization scheme that is more efficient than the layers in V1.
    """
    def __init__(self, d_input, d_output, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_input, d_output, bias=qkv_bias)
        self.W_key = nn.Linear(d_input, d_output, bias=qkv_bias)
        self.W_value = nn.Linear(d_input, d_output, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vector = attn_weights @ values
        return context_vector