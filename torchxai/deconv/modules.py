__author__ = "simonjisu"

import torch
import torch.nn as nn
from collections import OrderedDict

class DeconvNet(XaiModel):
    """DeconvNet"""
    def __init__(self, model, module_name="convs"):
        super(DeconvNet, self).__init__(model)
        layer_names = ["conv2d", "maxpool2d"]
        self.module_name = module_name
        self.deconvs_indices = self.find_idxes(module_name, layer_names)
        self.layers = self.make_layers(module_name)
        self.init_weights(module_name, layer_name="conv2d")
        
    def find_idxes(self, module_name, layer_names):
        """
        args:
        - module_name
        - layer_names: 
        
        return:
        get `deconvs_indices` for the `module_name`, 
        - key: decovnet layer name 
        - values: indices dict match to {convnet:decovnet}
        """
        convs_indices = self._find_target_layer_idx(module_name, layer_names)
        last_layer_num = len(self.model._modules[module_name]) - 1
        deconvs_indices = defaultdict(dict)
        
        for l_name in layer_names:
            idxes = (last_layer_num - torch.LongTensor(convs_indices[l_name])).tolist()
            deconvs_indices[l_name] = dict(zip(convs_indices[l_name], idxes))
            if l_name == "conv2d":
                deconvs_indices[l_name+"-bias"] = dict(zip(
                    convs_indices[l_name], idxes[1:]+[None]))
            
        return deconvs_indices
    
    def make_layers(self, module_name):
        """
        maxunpool > relu > conv 
        """
        layers = []
        modules = self.model._modules[module_name]
        for layer in modules:
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
                layers.append(temp_layer)
            elif isinstance(layer, nn.MaxPool2d):
                temp_layer = nn.MaxUnpool2d(layer.kernel_size,
                                            layer.stride,
                                            layer.padding)
                layers.append(temp_layer)
            else:
                layers.append(layer)
        return nn.Sequential(*reversed(layers))
    
    def init_weights(self, module_name, layer_name):
        convs = self.model._modules[module_name]
        conv_indices = self.deconvs_indices[layer_name]
        conv_bias_indices = self.deconvs_indices[layer_name+"-bias"]
        for i, layer in enumerate(convs):
            if type(layer).__name__.lower() == layer_name:
                # ex: 3 conv layers (conv, relu, maxpool)
                # 'conv2d': {0: 8, 3: 5, 6: 2}
                # 'conv2d-bias': {0: 5, 3: 2, 6: None}
                
                deconv_idx = conv_indices.get(i)
                weight = convs[i].weight.data
                self.layers[deconv_idx].weight.data = weight
                
                deconv_bias_idx = conv_bias_indices.get(i)
                if deconv_bias_idx is not None:
                    bias = convs[i].bias
                    self.layers[deconv_bias_idx].bias = bias
                
                            
    def get_attribution(self, x, targets):
        unpool_locations = self.deconvs_indices["maxpool2d"]
        unpool_locations = {v: k for k, v in unpool_locations.items()}
        convs = self.model._modules[self.module_name]

        switches = OrderedDict()
        self._reset_maps()
        self._return_indices(convs, on=True)
        for i, layer in enumerate(convs):
            if isinstance(layer, nn.MaxPool2d):
                x, switch = layer(x)
                switches[i] = switch
            else:
                x = layer(x)
                
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.MaxUnpool2d):
                j = unpool_locations[i]
                x = layer(x, switches[j])
            elif isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                layer_name = type(layer).__name__.lower() + f"-{i}"
                self._save_maps(layer_name, x.data)
            else:
                x = layer(x)
        self._return_indices(convs, on=False)
        x_ret = x.clone().detach().data
        return x_ret