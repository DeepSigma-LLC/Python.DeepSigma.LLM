import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["embedding_dim"])
        self.drop_emb = nn.Dropout(config["dropout"])
        self.transformer_blocks = nn.Sequential(
            * [DummyTransformerBlock(config) for _ in range(config["num_layers"])] # Placeholder transformer blocks
        )
        self.final_norm = DummyLayerNorm(config["embedding_dim"]) # Placeholder layer norm
        self.out_head = nn.Linear(config["embedding_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module): # Placeholder transformer block
    def __init__(self, config):
        super().__init__()

    def forward(self, x): # Does nothing. Just returns the input.
        return x

class DummyLayerNorm(nn.Module): # Placeholder layer norm
    def __init__(self, config):
        super().__init__()

    def forward(self, x): # Does nothing. Just returns the input.
        return x