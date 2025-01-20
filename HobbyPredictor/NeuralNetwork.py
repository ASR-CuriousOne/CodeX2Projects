from torch import nn
import torch

class HobbyPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=13,out_features=64),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=64,out_features=16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(in_features=16,out_features=3)
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.layers(x)