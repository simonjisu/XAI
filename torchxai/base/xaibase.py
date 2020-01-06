__author__ = "simonjisu"

import torch
import torch.nn as nn
from collections import OrderedDict

class XaiBase(nn.Module):
    def __init__(self):
        super(XaiBase, self).__init__()
        """
        - need to define XaiHook class to use
        - defalut hook_function is save (module, input, output) to (m, i, o)
          if you want to use hook function, change `hook_function` 
        """
        self._reset_maps()
    
    def _get_layer_name(self, layer, lower=True):
        """
        returns the name of a class
        """
        name = type(layer).__name__
        if not lower:
            return name
        return name.lower()

    def _reset_maps(self):
        self.maps = OrderedDict()
        
    def _save_maps(self, layer_name, x):
        self.maps[layer_name] = x    
        
    def _register(self, hooks, backward=False, hook_fn=None):
        """
        - need to define XaiHook class to use
        - defalut hook_function is save (module, input, output) to (m, i, o)
          if you want to use hook function, change `hook_function` 
        """
        if not isinstance(hooks, list):
            hooks = [hooks]
        for hook in hooks:
            hook.register_hook(backward=backward, hook_fn=hook_fn)
    
    def _register_forward(self, hooks, hook_fn=None):
        self._register(hooks, backward=False, hook_fn=hook_fn)
        
    def _register_backward(self, hooks, hook_fn=None):
        self._register(hooks, backward=True, hook_fn=hook_fn)
    
    def _reset_hooks(self, hooks):
        if not isinstance(hooks, list):
            hooks = [hooks]
        for hook in hooks:
            hook.close()

    def _return_indices(self, layers, on=True):
        """
        support for cnn layer which have `nn.MaxPool2d`,
        you can turn on/off pooling indices.
        please define a forward function to use it in your model
        '''
        # in your model
        def forward_switch(self, x):
            switches = OrderedDict()
            self.return_indices(on=True)
            for idx, layer in enumerate(self.convs):
                if isinstance(layer, nn.MaxPool2d):
                    x, indices = layer(x)
                    switches[idx] = indices
                else:
                    x = layer(x)
            self.return_indices(on=False)
            return x, switches
        '''
        """
        if on:
            for layer in layers:
                if isinstance(layer, nn.MaxPool2d):
                    layer.return_indices = True
        else:
            for layer in layers:
                if isinstance(layer, nn.MaxPool2d):
                    layer.return_indices = False  