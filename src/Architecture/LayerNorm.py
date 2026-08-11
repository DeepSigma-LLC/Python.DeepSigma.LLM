import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps # epsilon for numerical stability. Added to the variance to avoid dividing by zero.

        # trainable scaling factor. Training will adjust if needed to improve performance.
        self.scale = nn.Parameter(torch.ones(emb_dim))

        # trainable shift. Training will adjust if needed to improve performance.
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        # Enabling scale and shift to be trainable allows the model to learn the optimal scaling and shift values for the data being processed.

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # unbiased=False to match the original implementation. Matches tensorflow's implementation (originally used in GPT-2).
        # Determines if degrees of freedom are reduced by 1.

        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
