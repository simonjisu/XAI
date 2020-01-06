__author__ = "simonjisu"

import torch
import torch.nn as nn

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