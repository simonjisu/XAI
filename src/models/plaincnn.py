__author__ = "simonjisu"
__all__ = [
    "CnnMnist", "CnnMnistCBAM", "CnnMnistANR"
]

from torchxai.base import XaiBase
import torch
import torch.nn as nn
from torchxai.module import cbam, anr


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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def forward_map(self, x):
        """get activation maps"""
        self._reset_maps()
        for i, layer in enumerate(self.convs):
            layer_name = self._get_layer_name(layer)
            if layer_name == "relu":
                x, attns = layer(x, return_attn=True)
                self._save_maps(f"{i}"+layer_name, attns)
            else:
                x = layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CnnMnistCBAM(XaiBase):
    def __init__(self):
        super(CnnMnist, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # (B, 1, 28, 28) > (B, 32, 24, 24)
            cbam.CBAM(C=32, ratio=16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 24, 24) > (B, 32, 12, 12)
            nn.Conv2d(32, 64, 3),  # (B, 32, 12, 12) > (B, 64, 10, 10)
            cbam.CBAM(C=64, ratio=16, kernel_size=7, stride=1),
            nn.ReLU(), 
            nn.MaxPool2d(2),  # (B, 64, 10, 10) > (B, 64, 5, 5)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def register_attention_hooks(self):
        self.attn_hooks = OrderedDict()
        for name, m in self.named_modules():
            if isinstance(m, cbam.CBAM):
                c_attn_hook = XaiHook(m.channel_attn)
                s_attn_hook = XaiHook(m.spatial_attn)
                self.attn_hooks[f"{name}-c_attn"] = c_attn_hook
                self.attn_hooks[f"{name}-s_attn"] = s_attn_hook
        self._register_forward(list(self.attn_hooks.values()))

    def forward(self, x):        
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def close(self):
        """close all hooks"""
        self._reset_hooks(list(self.attn_hooks.values()))

    def forward_map(self, x):
        self.register_attention_hooks()
        self._reset_maps()
        o = self.forward(x)
        for k, hook in self.attn_hooks.items():
            self._save_maps(k, hook.o)
        return o


class CnnMnistANR(XaiBase):
    def __init__(self):
        super(CnnMnist, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # (B, 1, 28, 28) > (B, 32, 24, 24)
            anr.AttentionModule(in_c=32, n_head=4, n_label=10, reg_weight=0.01),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 24, 24) > (B, 32, 12, 12)
            nn.Conv2d(32, 64, 3),  # (B, 32, 12, 12) > (B, 64, 10, 10)
            anr.AttentionModule(in_c=32, n_head=4, n_label=10, reg_weight=0.01),
            nn.ReLU(), 
            nn.MaxPool2d(2),  # (B, 64, 10, 10) > (B, 64, 5, 5)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.global_attn_gate = anr.GlobalAttentionGate(in_c=64, n_hypothesis=3, gate_fn="softmax")
        
    def forward(self, x):
        reg_losses = 0.0
        hypothesis = []
        for layer in self.convs:
            if isinstance(layer, anr.AttentionModule):
                hypo = layer(x)
                reg_losses += layer.reg_loss()
                hypothesis.append(hypo.unsqueeze(1))  # + (B, 1, L)
            else:
                x = layer(x)
        
        last_conv = x
        x = torch.flatten(x, 1)
        x = self.fc(x)
        hypothesis.append(x.unsqueeze(1))

        o = self.global_attn_gate(last_conv, hypothesis)
        
        return o, reg_losses
    
    def register_attention_hooks(self):
        self.attn_hooks = OrderedDict()
        for name, m in self.named_modules():
            if isinstance(m, anr.AttentionModule):
                attn_hook = XaiHook(m.attn_heads)
                self.attn_hooks[f"{name}-attn"] = attn_hook
        self._register_forward(list(self.attn_hooks.values()))

    def close(self):
        """close all hooks"""
        self._reset_hooks(list(self.attn_hooks.values()))

    def forward_map(self, x):
        self.register_attention_hooks()
        self._reset_maps()
        o, reg_losses = self.forward(x)
        for k, hook in self.attn_hooks.items():
            self._save_maps(k, hook.o)
        return o, reg_losses