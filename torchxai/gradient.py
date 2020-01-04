__author__ = "simonjisu"

import torch
import torch.nn as nn
from .base import XaiModel, XaiHook

class VanillaGrad(XaiModel):
    """VanillaGrad"""
    def __init__(self, model):
        super(VanillaGrad, self).__init__(model)
        
    def get_attribution(self, x, targets):
        """vanilla gradient"""
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        output = self.model(x)
        grad = self._one_hot(targets, module_name="fc")
        output.backward(grad)
        x_grad = x.grad.data.clone()
        x.requires_grad_(requires_grad=False)
        return x_grad


class InputGrad(XaiModel):
    """InputGrad"""
    def __init__(self, model):
        super(InputGrad, self).__init__(model)
        
    def get_attribution(self, x, targets):
        """vanilla gradient*input"""
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        output = self.model(x)
        grad = self._one_hot(targets, module_name="fc")
        output.backward(grad)
        x_grad = x.grad.data.clone()
        x.requires_grad_(requires_grad=False)
        
        return x_grad * x.data


class GuidedGrad(XaiModel):
    """GuidedGrad"""
    def __init__(self, model):
        """
        applied at relu function
        """
        super(GuidedGrad, self).__init__(model)
        self.register_guided_hooks(self.model.convs)
    
    def reset_f_outputs(self):
        self.f_outputs = []
    
    def register_guided_hooks(self, layers):
        self.relu_f_hooks = []
        self.relu_b_hooks = []
        self.reset_f_outputs()
        for layer in layers:
            layer_name = type(layer).__name__
            if layer_name.lower() == "relu":
                f_hook = XaiHook(layer)
                b_hook = XaiHook(layer)
                self.relu_f_hooks.append(f_hook)
                self.relu_b_hooks.append(b_hook)
                
        def guided_forward(m, i, o):
            self.f_outputs.append(o.data)  
            
        def guided_backward(m, i, o):
            deconv_grad = o[0].clamp(min=0)  # o: backward input
            forward_output = self.f_outputs.pop(-1)
            forward_mask = forward_output.ne(0.0).type_as(forward_output)
            grad_in = deconv_grad * forward_mask
            return (grad_in, )
        
        # register forward hooks
        self._register_forward(self.relu_f_hooks, hook_fn=guided_forward)
        self._register_backward(self.relu_b_hooks, hook_fn=guided_backward)
        
    def get_attribution(self, x, targets):
        x.requires_grad_(True)
        output = self.model(x)
        grad = self._one_hot(targets, module_name="fc")
        output.backward(grad)
        x_grad = x.grad.data.clone()
        x.requires_grad_(False)
        return x_grad