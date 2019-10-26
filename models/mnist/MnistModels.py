__author__ = "simonjisu"
# models/mnist

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from reshape import Reshape

import torch
import torch.nn as nn
from collections import OrderedDict

class MNISTmodel(nn.Module):
    """
    paper implementation 'https://arxiv.org/abs/1711.06104'
    
    model_type: 
        - DNN: 
            Reshape: 28,28 > 28*28
            Linear: 28*28, 512 
            Linear: 512, 512
            Linear: 512, 10
        - CNN: 
            Conv2d: (1, 32, 3)
            Conv2d: (32, 64, 3)
            MaxPool2d: (2)
            Reshape: 12,12 > 12*12
            Linear: (64*12*12, 128)
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
        super(MNISTmodel, self).__init__()
        self.model_type = model_type
        self.activation_type = activation_type
        self.act = {"relu": nn.ReLU, 
                    "tanh": nn.Tanh, 
                    "sigmoid": nn.Sigmoid, 
                    "softplus": nn.Softplus}
        
        self.layers = self.make_layers(model_type, activation_type)

        self.activation_maps = OrderedDict()
        
    def make_layers(self, model_type, activation_type):
        self.activation_func = self.act[activation_type.lower()]
        if model_type.lower() == "dnn":
            layers = nn.Sequential(
                Reshape(),
                nn.Linear(28*28, 512),
                self.activation_func(), 
                nn.Linear(512, 512),
                self.activation_func(), 
                nn.Linear(512, 10),
            )
        elif model_type.lower() == "cnn":
            layers = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                self.activation_func(),
                nn.Conv2d(32, 64, 3),
                self.activation_func(), 
                nn.MaxPool2d(2),
                Reshape(),
                nn.Linear(64*12*12, 128),
                self.activation_func(),
                nn.Linear(128, 10)
            )
        else:
            assert False, "please insert `model_type` = `dnn` or `cnn`"
        return layers
    
    def save_activation_maps(self, layer, typ, idx, x):
        if isinstance(layer, typ):
            layer_name = f"({idx}) {str(layer).split('(')[0]}"
            
            self.activation_maps[layer_name] = x
    
    def forward(self, x, store=False):
        """
        store: if True, save activation maps
        """
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if store:
                self.save_activation_maps(layer, self.activation_func, idx, x)
        return x
