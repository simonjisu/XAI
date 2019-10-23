import torch
import torch.nn as nn
from collections import OrderedDict

class MNISTmodel(nn.Module):
    """
    paper implementation 'https://arxiv.org/abs/1711.06104'
    
    model_type: 
        - DNN: 
            Linear: 28*28, 512 
            Linear: 512, 512
            Linear: 512, 10
        - CNN: 
            Conv2d: (1, 32, 3)
            Conv2d: (32, 64, 3)
            MaxPool2d: (2)
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
        self.act = {"relu": nn.ReLU, 
                    "tanh": nn.Tanh, 
                    "sigmoid": nn.Sigmoid, 
                    "softplus": nn.Softplus}
        
        self.convs, self.fc = self.make_layers(model_type, activation_type)
        
        if model_type == "cnn":
            self.switches = OrderedDict()
            self.maxpool2d_locs = []
            self.convs_len = len(self.convs)
            
        self.activation_maps = OrderedDict()
        self.activation_locs = []
        self.fc_len = len(self.fc)
        
    def make_layers(self, model_type, activation_type):
        self.activation_func = self.act[activation_type.lower()]
        if model_type.lower() == "dnn":
            convs = None
            fc = nn.Sequential(
                nn.Linear(28*28, 512),
                self.activation_func(), 
                nn.Linear(512, 512),
                self.activation_func(), 
                nn.Linear(512, 10),
            )
        elif model_type.lower() == "cnn":
            convs = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                self.activation_func(),
                nn.Conv2d(32, 64, 3),
                self.activation_func(), 
                nn.MaxPool2d(2, return_indices=True),
            )
            fc = nn.Sequential(
                nn.Linear(64*12*12, 128),
                self.activation_func(),
                nn.Linear(128, 10)
            )
        else:
            assert False, "please insert `model_type` = `dnn` or `cnn`"
        return convs, fc    
    
    def save_activation_maps(self, layer, idx, x, typ, indices=None):
        if isinstance(layer, self.activation_func):
            layer_name = f"{typ}{idx}"
            self.activation_locs.append(layer_name)
            self.activation_maps[layer_name] = x
    
    def forward(self, x, store=False):
        # conv layers
        if self.convs is not None:
            for idx, layer in enumerate(self.convs):
                if isinstance(layer, nn.MaxPool2d):
                    x, indices = layer(x)
                    self.switches[idx] = indices
                    self.maxpool2d_locs.append(idx)
                else:
                    x = layer(x)
                    if store:
                        self.save_activation_maps(layer, idx, x, typ="convs")
        # resize
        x = x.view(x.size(0), -1)
        # fc layers
        for idx, layer in enumerate(self.fc, (self.convs_len+self.fc_len-2)*int(bool(self.convs))):
            x = layer(x)
            if store:
                self.save_activation_maps(layer, idx, x, typ="fc")
        return x