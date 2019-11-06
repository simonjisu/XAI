__author__ = "simonjisu"
# models/deconv

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from reshape import Reshape


import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy

class deconvMNIST(nn.Module):
    """deconvMNIST"""
    def __init__(self, model, load_path=None):
        """
        do not load model parameters first
        """
        super(deconvMNIST, self).__init__()
        assert load_path, "insert `load_path` model"
        # deconv
        self.activation_func = model.activation_func
        self.model_type = model.model_type
        self.activation_type = model.activation_type
        
        model.load_state_dict(torch.load(load_path, map_location="cpu"))
        self.layers = self.deconv_make_layers(model)
        
        self.activation_maps = OrderedDict()
        
    def deconv_make_layers(self, model):
        layers = []
        
        conv_pos = {}  # {deconv : conv}
        conv_locs = [i for i, l in enumerate(model.layers) if isinstance(l, nn.Conv2d)]
        # [0, 3]
        # {6: 0, 999: 3}
        for idx, i in enumerate(conv_locs):
            if idx == (len(conv_locs) - 1):
                conv_pos[999] = i
            else:
                j = len(model.layers) - conv_locs[idx+1] -1
                conv_pos[j] = i 
        conv_pos_reverse = {v: k for k, v in conv_pos.items()}
        for idx, layer in enumerate(model.layers[::-1]):
            if isinstance(layer, nn.Linear):
                pass
#                 temp_layer = nn.Linear(layer.out_features, layer.in_features, bias=False)
#                 temp_layer.weight.data = layer.weight.T.data
#                 temp_layer.bias.data = layer.bias.data
#                 layers.append(temp_layer)
            elif isinstance(layer, nn.Conv2d):
                temp_layer = nn.ConvTranspose2d(layer.out_channels,
                                                layer.in_channels,
                                                layer.kernel_size, 
                                                layer.stride, 
                                                layer.padding,
                                                layer.output_padding,
                                                layer.groups, 
                                                False,  # bias
                                                layer.dilation,
                                                layer.padding_mode)
                temp_layer.weight.data = layer.weight.data
                # add bias to previous layer, 
                # cannot add last conv layer, size doesn't match
                check_idx = len(model.layers) - 1 - idx
                if conv_pos_reverse[check_idx] != 999:  # if it is not last conv
                    temp_layer.bias.data = model.layers[conv_pos[check_idx]]
                layers.append(temp_layer)
            elif isinstance(layer, nn.MaxPool2d):
                temp_layer = nn.MaxUnpool2d(layer.kernel_size,
                                            layer.stride,
                                            layer.padding)
                layers.append(temp_layer)
            else:
                layers.append(layer)
                
        return nn.Sequential(*layers)
    
    def save_activation_maps(self, layer, typ, idx, x):
        if isinstance(layer, typ):
            layer_name = f"({idx}) {str(layer).split('(')[0]}"
            self.activation_maps[layer_name] = x
    
    def deconv(self, x, switches, store=False):
        """
        store: if True, save activation maps
        """
        switches = OrderedDict()
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                switches[idx] = indices
            else:
                x = layer(x)
        
        o = x
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.MaxUnpool2d):
                o = layer(o, switches[len(self.layers)-1-idx])
            elif isinstance(layer, Reshape):
                o = layer(o, backward=True)
            else:
                o = layer(o)
                if store:
                    self.save_activation_maps(layer, nn.ConvTranspose2d, idx, o)
        return o