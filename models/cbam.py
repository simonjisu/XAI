__author__ = "simonjisu"

import torch
import torch.nn as nn
from collections import OrderedDict

class CnnWithCBAM(XaiBase):
    def __init__(self):
        super(CnnWithCBAM, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # (B, 1, 28, 28) > (B, 32, 24, 24)
            CBAM(32, 24, 24, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 24, 24) > (B, 32, 12, 12)
            nn.Conv2d(32, 64, 3),  # (B, 32, 12, 12) > (B, 64, 10, 10)
            CBAM(64, 10, 10, 16),
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
            layer_name = type(layer).__name__.lower()
            if layer_name == "cbam":
                x, attns = layer(x, return_attn=True)
                self._save_maps(f"{i}"+layer_name, attns)
            else:
                x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x