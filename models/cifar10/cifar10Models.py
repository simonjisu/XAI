__author__ = "simonjisu"
# models/cifar10

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from reshape import Reshape

import torch
import torch.nn as nn
from collections import OrderedDict

class Cifar10model(nn.Module):
    """
    paper implementation 'https://arxiv.org/abs/1711.06104'
    
    model_type: 
        - DNN: 
            Reshape: 3,32,32 > 3*32*32
            Linear: 3*32*32, 1024 
            Linear: 1024, 512
            Linear: 512, 10
        - CNN: 
            Conv2d: (3, 16, 5)      32 > 28
            MaxPool2d: (2)          28 > 14
            Conv2d: (16, 32, 3)     14 > 12
            MaxPool2d: (2)          12 > 6
            Conv2d: (32, 64, 3)      6 > 4
            MaxPool2d: (2)           4 > 2
            Reshape: 2,2 > 2*2 
            Linear: (64*2*2, 128)
            Linear: (128, 10)

            ** Conv2d = (in_kernels, out_kernels, kernel_size)
            ** MAxPool2d = (kernel_size)

    activation_type:
        - ReLU, Tanh, Sigmoid, Softplus
    """
    def __init__(self, model_type, activation_type):
        """
        model_type: "dnn", "cnn"
        activation_type: "relu", "tanh", "sigmoid", "softplus"
        """
        super(Cifar10model, self).__init__()
        self.model_type = model_type
        self.activation_type = activation_type
        self.act = {"relu": nn.ReLU, 
                    "tanh": nn.Tanh, 
                    "sigmoid": nn.Sigmoid, 
                    "softplus": nn.Softplus}
        
        self.layers = self.make_layers(model_type, activation_type)
        
    def reset_activation_maps(self):
        self.activation_maps = OrderedDict()
        
    def make_layers(self, model_type, activation_type):
        self.activation_func = self.act[activation_type.lower()]
        if model_type.lower() == "dnn":
            layers = nn.Sequential(
                Reshape(),
                nn.Linear(3*32*32, 1024),
                self.activation_func(), 
                nn.Linear(1024, 512),
                self.activation_func(), 
                nn.Linear(512, 10),
            )
        elif model_type.lower() == "cnn":
            layers = nn.Sequential(
                nn.Conv2d(3, 16, 5),
                nn.BatchNorm2d(16),
                self.activation_func(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(16, 32, 3),
                nn.BatchNorm2d(32),
                self.activation_func(), 
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 3),
                nn.BatchNorm2d(64),
                self.activation_func(), 
                nn.MaxPool2d(2),
                Reshape(),
                nn.Linear(64*2*2, 128),
                self.activation_func(),
                nn.Linear(128, 10)
            )
        else:
            assert False, "please insert `model_type` = `dnn` or `cnn`"
        return layers
    
    def return_indices(self, on=True):
        if on:
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    layer.return_indices = True
        else:
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    layer.return_indices = False
    
    def save_activation_maps(self, layer, typ, idx, x):
        if isinstance(layer, typ):
            if typ == self.activation_func:
                layer_name = f"({idx}) {str(self.layers[idx-1]).split('(')[0]}>{self.activation_type}"
            else:
                layer_name = f"({idx}) Conv2d>{self.activation_type}>MaxPool2d"
            self.activation_maps[layer_name] = x.detach()
    
    def forward(self, x, store=False):
        """
        store: if True, save activation maps
        """
        self.reset_activation_maps()
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if store:
                self.save_activation_maps(layer, self.activation_func, idx, x)
        return x
    
    def forward_switches(self, x, store=False):
        """
        get max pool indices & store activation maps
        
        output: 
            - convs output before reshape to fc
            - switches
        """
        switches = OrderedDict()
        self.reset_activation_maps()
        self.return_indices(on=True)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                switches[idx] = indices
                if store:
                    self.save_activation_maps(layer, nn.MaxPool2d, idx, x)
            elif isinstance(layer, Reshape):
                break
            else:
                x = layer(x)
        self.return_indices(on=False)
        return x, switches