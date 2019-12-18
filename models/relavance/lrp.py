__author__ = "simonjisu"
# models/relavance

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from reshape import Reshape

import torch
import torch.nn as nn
from collections import OrderedDict
from .layers import relLinear, relConv2d, relMaxPool2d, relReLU

class LRP(nn.Module):
    """LRP"""
    def __init__(self, model):
        super(LRP, self).__init__()
        # lrp
        self.activation_func = model.activation_func
        self.model_type = model.model_type
        self.activation_type = model.activation_type
        
        self.layers = self.lrp_make_layers(model)
        
    def reset_activation_maps(self):
        self.activation_maps = OrderedDict()

    def lrp_make_layers(self, model):
        layers = []
        mapping_dict = {nn.Linear: relLinear, nn.Conv2d: relConv2d, nn.MaxPool2d: relMaxPool2d, 
                        nn.ReLU: relReLU}
        for layer in model.layers:
            if isinstance(layer, Reshape):
                layers.append(layer)
            else:
                layers.append(mapping_dict[layer.__class__](layer))
                
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        lrp method
        must run forward first to save input and output at each layer
        """
        self.reset_activation_maps()
        for layer in self.layers:
            x = layer(x)
        return x
    
    def save_activation_maps(self, layer, typ, idx, x):
        if isinstance(layer, typ):
            layer_name = f"({idx}) {str(layer).split('(')[0]}"
            self.activation_maps[layer_name] = x
    
    def get_attribution(self, x, target=None, store=False, use_rho=False):
        """
        store: if True, save activation maps
        """
        o = self.forward(x).detach()
        r = o * torch.zeros_like(o).scatter(1, o.argmax(1, keepdim=True), 1)
        for idx, layer in enumerate(self.layers[::-1]):
            r = layer.relprop(r, use_rho)
            if store:
                self.save_activation_maps(layer, relConv2d, idx, r)
        return r.detach()