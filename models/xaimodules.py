import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict

class XaiBase(nn.Module):
    def __init__(self):
        super(XaiBase, self).__init__()
        """
        need to define hook function at each method
        - f_hook
        - b_hook
        """
        self._reset_maps()
        self.handlers = list()
    
    def _reset_maps(self):
        self.maps = OrderedDict()
        
    def _save_maps(self, layer_name, x):
        self.maps[layer_name] = x    
    
    def _reset_handlers(self):
        for handle in self.handlers:
            handle.remove()
        self.handlers = []
                
    def _register(self, registor_type="both"):
        """
        need to define hook functions to use
         - both: registor both forward and backward
         - f: registor forward
         - b: registor backward
        
        def f_hook(self, *x):
            '''
            m: module name
            i: forward input
            o: forward output
            '''
            m, i, o = x
        
        def b_hook(self, *x):
            '''
            m: module name
            i: gradient input
            o: gradient output
            '''
            m, i, o = x
        """
        for layer in self.model.layers:
            if registor_type == "both":
                handle1 = layer.register_forward_hook(self.f_hook)
                handle2 = layer.register_backward_hook(self.b_hook)
                self.handlers.append(handle1)
                self.handlers.append(handle2)
            elif registor_type == "f":
                handle1 = layer.register_forward_hook(self.f_hook)
                self.handlers.append(handle1)
            elif registor_type == "b":
                handle2 = layer.register_backward_hook(self.b_hook)
                self.handlers.append(handle2)

    def _return_indices(self, on=True):
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
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    layer.return_indices = True
        else:
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    layer.return_indices = False  
                    
                    
class XaiModel(XaiBase):
    def __init__(self, model):
        super(XaiModel, self).__init__()
        self.model = deepcopy(model)