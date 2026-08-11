import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, config, expansion_factor=4):
        """
        FeedForward network used to transform the embedding vector.
        :param config: The config dictionary containing the model's hyperparameters.
        :param expansion_factor: Used to increase the internal representation size of the feedforward network.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], expansion_factor * config["embedding_dim"]),
            nn.GELU(),
            nn.Linear( expansion_factor * config["embedding_dim"], config["embedding_dim"])
        )

    def forward(self, x):
        return self.layers(x)