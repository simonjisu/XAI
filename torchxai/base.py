import torch
import torch.nn as nn
from copy import deepcopy
from collections import defaultdict, OrderedDict

class XaiHook(nn.Module):
    def __init__(self, module):
        super(XaiHook, self).__init__()
        """
        Hook Handler Module
        
        supported register `module` hooks
        - Activations
        - Linear
        - Convd
        
        like RNN have to use `register_hook` to `torch.nn.Parameter` directly
        
        * Ref: https://pytorch.org/docs/master/nn.html#torch.nn.Module.register_backward_hook
        [Warnings]
        The current implementation will not have the presented behavior 
        for complex Module that perform many operations. In some failure cases, 
        `grad_input` and `grad_output` will only contain the gradients for a subset
        of the inputs and outputs. For such `Module`, you should use 
        `torch.Tensor.register_hook()` directly on a specific input or 
        output to get the required gradients.
        
        """
        self.module = module
    
    def forward(self, x):
        return self.module.forward(x)

    def zero_grad(self):
        self.module.zero_grad()

    def register_hook(self, backward=False, hook_fn=None):
        """
        defalut hook_function is save (module, input, output) to (m, i, o)
        if you want to use hook function, change `hook_function` 
        if `hook_function` returns `None` then the original input or output 
        will be flow into next / previous layer, but you can return a modifed
        output/gradient to change the original output/gradient.
        for a Conv2d layer example
        - forward: a `Tensor` type output
        - backward: (gradient_input, weight, bias)
        
        """
        def default_hook_fn(m, i, o):
            """
            forward
             - m: module class
             - i: forward input from previous layer
             - o: forward output to next layer
            backward
             - m: module class
             - i: gradient input to next layer (backward out)
             - o: gradient output from previous layer (backward in)

            args:
             * i, o: tuple type
            """
            self.m = m
            self.i = i
            self.o = o
            
        if hook_fn is None:
            self.hook_fn = default_hook_fn
        else:
            self.hook_fn = hook_fn
        if not backward:
            self.hook = self.module.register_forward_hook(self.hook_fn)
        else:
            self.hook = self.module.register_backward_hook(self.hook_fn)
            
    def close(self):
        self.hook.remove()


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
                    
                    
class XaiModel(XaiBase):
    def __init__(self, model):
        super(XaiModel, self).__init__()
        self.model = deepcopy(model)
        self.model.cpu()
        self.model.eval()
        
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
            l_name = type(layer).__name__.lower()
            if l_name in layer_names:
                idxes[l_name].append(int(idx))

        return idxes