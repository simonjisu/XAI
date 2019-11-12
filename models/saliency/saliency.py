import torch
import torch.nn as nn
from copy import deepcopy

class saliencyMNIST(nn.Module):
    def __init__(self, model):
        """
        do not load model parameters first
        """
        super(saliencyMNIST, self).__init__()
        
        # vanilla saliency
        self.activation_func = model.activation_func
        self.model_type = model.model_type
        self.activation_type = model.activation_type

        self.model = model
        self.model.eval()
        
    def get_attribution(self, x, target):
        """vanilla gradient*input"""
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        o = self.model(x)
        grad_outputs = torch.zeros_like(o).scatter(1, target.unsqueeze(1), 1).detach()
        o.backward(gradient=grad_outputs)
        x.requires_grad_(requires_grad=False)
        
        return x.grad.clone() * x