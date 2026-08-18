import math
import torch


class LoRALayer(torch.nn.Module):
    """
    LoRA layer implementation.
    The rank governs the inner dimension of matrices A and B. Essentially, this setting
    determines the number of extra parameters introduced by LoRA, which creates balance between the adaptability of
    the model and its efficiency via the number of parameters used.

    The other important setting, alpha, functions as a scaling factor for the output from the low-rank adaptation.
    It primarily dictates the degree to which the output from the adapted layer can affect the original layer’s output.
    """
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        x = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return x