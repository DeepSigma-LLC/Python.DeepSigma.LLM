import torch
from src.Architecture.LoRALayer import LoRALayer

class LinearWithLoRA(torch.nn.Module):
    """
    To integrate the original Linear layer weights, we now create a LinearWithLoRA layer.
    This layer utilizes the previously implemented LoRALayer and is designed to replace
    existing Linear layers within a neural network, such as the self-attention modules or
    feed-forward modules in the GPTModel.
    """
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)