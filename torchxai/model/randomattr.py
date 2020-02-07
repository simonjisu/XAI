__author__ = "simonjisu"

import torch
import torch.nn as nn
from ..base import XaiModel

class Random(XaiModel):
    """VanillaGrad"""
    def __init__(self, model, norm_mode=1, abs_grad=False):
        super(Random, self).__init__(model)
        self.norm_mode = 1
        
    def get_attribution(self, x, seed):
        """vanilla gradient"""
        torch.manual_seed(seed)
        B, C, H, W = x.size()
        o = torch.rand(B, C*H*W).view(B, C, H, W)
        if self.norm_mode:
            o = self._normalization(o, norm_mode=self.norm_mode)
        return o
