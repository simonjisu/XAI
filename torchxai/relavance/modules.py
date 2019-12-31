__author__ = "simonjisu"

import torch
import torch.nn as nn
from collections import OrderedDict
from .layers import relLinear, relConv2d, relMaxPool2d, relReLU
from ..base import XaiModel

class LRP(XaiModel):
    """LRP"""
    def __init__(self, model, use_rho=False):
        """
        module_names: have to be sequential to forward network 
        """
        super(LRP, self).__init__(model)
        self.module_names = module_names
        self.use_rho = use_rho
        self.available_module = {
            nn.Linear: relLinear, 
            nn.Conv2d: relConv2d, 
            nn.MaxPool2d: relMaxPool2d, 
            nn.ReLU: relReLU
        }
        self.create_layers()
        
    def create_layers(self):
        for name, modules in self.model._modules.items():
            if isinstance(modules, nn.Sequential):
                for i, layer in enumerate(modules):
                    try:
                        modules[i] = self._create_layer(layer)
                    except KeyError as e:
                        print(f"{type(layer)} is not an available module.\nAvaiable:")
                        for k in self.available_module.keys():
                            print(f" - {k}")
            else:
                modules = self._create_layer(layer)      
    
    def forward(self, x):
        return self.model(x)
    
    def _create_layer(self, layer):
        if isinstance(layer, nn.ReLU):
            return self.available_module[type(layer)](layer)
        else:
            return self.available_module[type(layer)](layer, use_rho=self.use_rho)

    def get_attribution(self, x, targets):
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        output = self.forward(x)
        grad = self._one_hot(targets, module_name="fc")
        output.backward(grad)
        x_grad = x.grad.data.clone().detach()
        x.requires_grad_(requires_grad=False)
        return x_grad