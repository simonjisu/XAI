__author__ = "simonjisu"
# models/deconv

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from reshape import Reshape


import torch
import torch.nn as nn
from collections import OrderedDict

class DeconvNet(nn.Module):
    """DeconvNet"""
    def __init__(self, model):
        super(DeconvNet, self).__init__()
        # deconv
        self.activation_func = model.activation_func
        self.model_type = model.model_type
        self.activation_type = model.activation_type
        
        self.layers = self.deconv_make_layers(model)
        
        self.activation_maps = OrderedDict()
        
    def deconv_make_layers(self, model):
        self.deconv_module_len = 0
        layers = []
        conv_end = [i for i, l in enumerate(model.layers) if str(l) == "Reshape()"][0]
        # {0: 999, 3: 0}
        conv_bias_pos = {}
        conv_locs = [i for i, l in enumerate(model.layers[:conv_end]) if isinstance(l, nn.Conv2d)]
        for idx, i in enumerate(conv_locs):
            if idx == 0:
                conv_bias_pos[i] = 999
            else:
                conv_bias_pos[i] = conv_locs[idx-1]

        for idx, layer in enumerate(model.layers[:conv_end]):
            if isinstance(layer, nn.Conv2d):
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
                if conv_bias_pos[idx] < 999:
                    temp_layer.bias = model.layers[:conv_end][conv_bias_pos[idx]].bias
                layers.append(temp_layer)
            elif isinstance(layer, nn.MaxPool2d):
                temp_layer = nn.MaxUnpool2d(layer.kernel_size,
                                            layer.stride,
                                            layer.padding)
                layers.append(temp_layer)
                self.deconv_module_len += 1
            else:
                layers.append(layer)
        layers = nn.Sequential(*reversed(layers))
        
        deconv_locs = [i for i, l in enumerate(layers) if isinstance(l, nn.ConvTranspose2d)]
        self.conv_end = conv_end
        # {2: 2, 1: 5}
        self.deconv_locs = {(len(deconv_locs) - j):i for j, i in enumerate(deconv_locs)}
        return layers
    
    def save_activation_maps(self, layer, typ, idx, x):
        if isinstance(layer, typ):
            layer_name = f"({idx}) {str(layer).split('(')[0]}(in:{layer.in_channels}, out:{layer.out_channels})"
            self.activation_maps[layer_name] = x
    
    def deconv(self, x, switches, deconv_layer_num=None, store=False):
        """
        deconv_layer_num: 
            deconv from which module(m =  "MaxPool > activation > Conv2d") 
            numbering from the original cnn conv module(n = "Conv2d > activation > MaxPool")
            ex) deconv_layers = [m1, m2, m3, m4, m5]
                if deconv_layer_num = 4, will goes from m4 to m5
                * cnn_layer = [n5, n4, n3, n2, n1] (n.T = m, n5 is the first layer of cnn)
                
        x: should match module input size
        switches: from MNISTmodel forward method "forward_switches"
        store: if True, save activation maps
        """
        assert (deconv_layer_num <= self.deconv_module_len) or (deconv_layer_num==None), \
            "`deconv_layer_num` should <= `self.deconv_module_len` or == None"
        if deconv_layer_num == None: deconv_layer_num = 1
        deconvfrom = self.deconv_locs[deconv_layer_num]
        deconvlayers = self.layers[-(deconvfrom+1):]
        unpool_locs = {idx:(len(deconvlayers)-1 - idx) for idx, l in enumerate(deconvlayers) if isinstance(l, nn.MaxUnpool2d)}
        
        for idx, layer in enumerate(deconvlayers):
            if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, switches[unpool_locs[idx]])
            else:
                x = layer(x)
                if store:
                    self.save_activation_maps(layer, nn.ConvTranspose2d, idx, x)
        return x