__author__ = "simonjisu"

import torch
import torch.nn as nn
from ..base import XaiModel, XaiHook

class VanillaGrad(XaiModel):
    """VanillaGrad"""
    def __init__(self, model, **kwargs):
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
    def __init__(self, model, **kwargs):
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

class GuidedReLU(XaiHook):
    """GuidedReLU"""
    def __init__(self, module):
        super(GuidedReLU, self).__init__(module)
        self.register_hook(backward=False)
        self.register_hook(backward=True, hook_fn=self.b_hook)

    def b_hook(self, m, i, o):
        """
        backward hook
        i: (input,) -> backward output
        o: (output,) -> backward input
        """
        deconv_grad = o[0].clamp(min=0)  # o: backward input
        forward_output = self.o
        forward_mask = forward_output.ne(0.0).type_as(forward_output)
        grad_in = deconv_grad * forward_mask
        return grad_in


class GuidedGrad(XaiModel):
    """GuidedGrad"""
    def __init__(self, model, module_name, act=nn.ReLU, **kwargs):
        """
        """
        super(GuidedGrad, self).__init__(model)
        self.act = act
        if not isinstance(module_name, list):
            module_name = [module_name]
        self.all_hooks = []
        for n_name in module_name:
            self.register_guided_hooks(n_name)

    def register_guided_hooks(self, n_name):
        modules = self.model._modules[n_name]
        if isinstance(modules, nn.Sequential):
            for i, layer in enumerate(modules):
                if isinstance(layer, self.act):
                    # change layer to Guided Layer
                    modules[i] = GuidedReLU(layer)
                    self.all_hooks.append(modules[i])
        else:
            if isinstance(modules, self.act):
                self.model._modules[n_name] = GuidedGrad(modules)


    def get_attribution(self, x, targets):
        x.requires_grad_(True)
        output = self.model(x)
        grad = self._one_hot(targets, module_name="fc")
        output.backward(grad)
        x_grad = x.grad.data.clone()
        x.requires_grad_(False)
        return x_grad


# class GuidedGrad(XaiModel):
#     """GuidedGrad"""
#     def __init__(self, model, module_name="convs", act=nn.ReLU, **kwargs):
#         """
#         applied at relu function
#         """
#         super(GuidedGrad, self).__init__(model)
#         self.act = act
#         if not isinstance(module_name, list):
#             module_name = [module_name]
#         self.f_hooks = []
#         self.b_hooks = []
#         for n_name in module_name:
#             self.register_guided_hooks(self.model._modules[n_name])
    
#     def reset_f_outputs(self):
#         self.f_outputs = []
    
#     def register_guided_hooks(self, layers):
#         if not isinstance(layers, nn.Sequential):
#             layers = [layers]
#         self.reset_f_outputs()
#         for layer in layers:
#             if isinstance(layer, self.act):
#                 f_hook = XaiHook(layer)
#                 b_hook = XaiHook(layer)
#                 self.f_hooks.append(f_hook)
#                 self.b_hooks.append(b_hook)
                
#         def guided_forward(m, i, o):
#             self.f_outputs.append(o.data)
            
#         def guided_backward(m, i, o):
#             deconv_grad = o[0].clamp(min=0)  # o: backward input
#             forward_output = self.f_outputs.pop(-1)
#             forward_mask = forward_output.ne(0.0).type_as(forward_output)
#             grad_in = deconv_grad * forward_mask
#             return (grad_in, )
        
#         # register forward hooks
#         self._register_forward(self.f_hooks, hook_fn=guided_forward)
#         self._register_backward(self.b_hooks, hook_fn=guided_backward)
        
#     def get_attribution(self, x, targets):
#         x.requires_grad_(True)
#         output = self.model(x)
#         grad = self._one_hot(targets, module_name="fc")
#         output.backward(grad)
#         x_grad = x.grad.data.clone()
#         x.requires_grad_(False)
#         return x_grad