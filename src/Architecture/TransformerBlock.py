import torch
import torch.nn as nn
from src.Architecture.FeedForward import FeedForward
from src.Attention.MultiHeadAttention import MultiHeadAttention
from src.Architecture.LayerNorm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config["embedding_dim"],
            d_out=config["embedding_dim"],
            context_length=config["context_length"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            qkv_bias=config["qkv_bias"])
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["embedding_dim"])
        self.norm2 = LayerNorm(config["embedding_dim"])
        self.dropout = nn.Dropout(config["dropout"])


    def forward(self, x):
        """
        Note: We use pre-LayerNorm rather than post-LayerNorm. This is because post-LayerNorm generally leads to worse training dynamics.
        This is common practice in modern transformer architectures.
        We also use residual connections across the two layers to improve gradient flow and stabilize training.
        :param x:
        :return:
        """
        shortcut = x        # store the input for residual connection
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut    # add the residual connection value

        shortcut = x        # store the new input for residual connection
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut    # add the residual connection value
        return x