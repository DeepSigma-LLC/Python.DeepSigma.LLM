import torch
import torch.nn as nn


class SelfAttentionV1(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_input, d_output))
        self.W_key = nn.Parameter(torch.rand(d_input, d_output))
        self.W_value = nn.Parameter(torch.rand(d_input, d_output))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vector = attn_weights @ values
        return context_vector