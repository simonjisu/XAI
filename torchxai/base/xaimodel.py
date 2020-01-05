__author__ = "simonjisu"

import torch
import torch.nn as nn
from .xaibase import XaiBase
from copy import deepcopy
from collections import defaultdict

class XaiModel(XaiBase):
    def __init__(self, model):
        super(XaiModel, self).__init__()
        self.model = deepcopy(model)
        self.model.cpu()
        self.model.eval()
    
    def get_attribution(self, *args):
        """
        all XaiModel should have `get_attribution` method
        """
        raise NotImplementedError

    def _one_hot(self, targets, module_name):
        """
        one hot vectorize the target tensor for classification purpose.
        the `module` with respect to `module_name` must have `out_features` attribution.
        args:
        - targets: torch.LongTensor, target classes that have size of mini-batch
        - module_name: str, feature name for Fully-Connected Network or any Task-specific Network
        return:
        - one hot vector of targets
        """
        assert isinstance(targets, torch.LongTensor), "`targets` must be `torch.LongTensor` type"
        assert isinstance(module_name, str), "`module_name` must be `str` type"
        modules = self.model._modules[module_name]
        if isinstance(modules, nn.Sequential):
            last_layer = modules[-1]
        else:
            last_layer = modules
        try:
            last_layer.out_features 
        except AttributeError as e:
            is_linear = isinstance(last_layer, nn.Linear)
            print(f"last layer of module `{module_name}` doesn't have `out_features` attribute")
            print()
            if not is_linear:
                print(f"type of the last layer is `{type(last_layer)}`")
                print("the last layer is not `torch.nn.linear.Linear` class")
                print("create `.out_featrues` attribution in the custom module")
                
        target_size = last_layer.out_features
        B = targets.size(0)
        one_hot = torch.zeros((B, target_size))
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        return one_hot.to(targets.device)
    
    def _find_target_layer_idx(self, module_name, layer_names):
        assert isinstance(layer_names, list) or isinstance(layer_names, tuple), "use list for `layer_names`"
        layer_names = [l.lower() for l in layer_names]
        idxes = defaultdict(list)
        modules = self.model._modules[module_name]
        assert isinstance(modules, nn.Sequential), "use this function for `nn.Sequential` type modules"
        for idx, layer in modules.named_children():
            # TODO [RF] 0.1.2 features/layername
            # Change `layer_names` to module classes `[nn.Conv2d, nn.Maxpool2d]`
            # so that use `isinstance` to check if target layer is in the modules
            l_name = type(layer).__name__.lower()
            if l_name in layer_names:
                idxes[l_name].append(int(idx))

        return idxes