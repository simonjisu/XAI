__author__ = "simonjisu"
__all__ = [
    "CnnMnist"
]

from torchxai.base import XaiBase
import torch
import torch.nn as nn

class CnnMnist(XaiBase):
    def __init__(self):
        super(CnnMnist, self).__init__()
        """
        CNN: 
        convs:
            Conv2d(1, 32, 5)      28 > 24
            ReLU()
            MaxPool2d(2)          24 > 12
            Conv2d(32, 64, 3)     12 > 10
            ReLU()
            MaxPool2d(2)          10 > 5
        fc:
            Linear(64*5*5, 128)
            Linear(128, 10)
        """
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # (B, 1, 28, 28) > (B, 32, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 24, 24) > (B, 32, 12, 12)
            nn.Conv2d(32, 64, 3),  # (B, 32, 12, 12) > (B, 64, 10, 10)
            nn.ReLU(), 
            nn.MaxPool2d(2),  # (B, 64, 10, 10) > (B, 64, 5, 5)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):        
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward_map(self, x):
        self._reset_maps()
        for i, layer in enumerate(self.convs):
            layer_name = self._get_layer_name(layer)
            if layer_name == "relu":
                x, attns = layer(x, return_attn=True)
                self._save_maps(f"{i}"+layer_name, attns)
            else:
                x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x