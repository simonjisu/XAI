import torch
import torch.nn as nn
from ..base import XaiModel

class VanillaGrad(XaiModel):
    """VanillaGrad"""
    def __init__(self, model):
        super(VanillaGrad, self).__init__(model)
        
    def get_attribution(self, x, target):
        """vanilla gradient"""
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        output = self.model(x)
        grad = self._one_hot(targets, module_name="fc")
        output.backward(grad)
        x_grad = x.grad.data.clone()
        x.requires_grad_(requires_grad=False)
        return x_grad


class GradInput(XaiModel):
    """GradInput"""
    def __init__(self, model):
        super(GradInput, self).__init__(model)
        
    def get_attribution(self, x, target):
        """vanilla gradient*input"""
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        output = self.model(x)
        grad = self._one_hot(targets, module_name="fc")
        output.backward(grad)
        x_grad = x.grad.data.clone()
        x.requires_grad_(requires_grad=False)
        
        return x_grad * x.data