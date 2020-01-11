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
        return (grad_in, )


class GuidedGrad(XaiModel):
    """GuidedGrad"""
    def __init__(self, model, act=nn.ReLU):
        """
        """
        super(GuidedGrad, self).__init__(model)
        self.act = act
        self.act_dict = {
            nn.ReLU: GuidedReLU
        }
        self.register_guided_hooks()

    def register_guided_hooks(self):
        self.all_hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, self.act):
                guided_module = self.act_dict[self.act](module)
                splited_name = name.split(".")
                prefix, layer_name = splited_name[:-1], splited_name[-1]
                modules = self.model._modules
                if len(prefix) >= 1:
                    for n in prefix:
                        modules = modules.get(n)._modules
                    modules[layer_name] = guided_module
                else:
                    modules[layer_name] = guided_module
                self.all_hooks.append(guided_module)

    def get_attribution(self, x, targets):
        x.requires_grad_(True)
        output = self.model(x)
        grad = self._one_hot(targets, module_name="fc")
        output.backward(grad)
        x_grad = x.grad.data.clone()
        x.requires_grad_(False)
        return x_grad