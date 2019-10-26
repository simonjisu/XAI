import torch
import torch.nn as nn
from copy import deepcopy

class saliencyMNIST(nn.Module):
    def __init__(self, model, load_path=None):
        """
        do not load model parameters first
        """
        super(saliencyMNIST, self).__init__()
        assert load_path, "insert `load_path` model"        
        
        # vanilla saliency
        self.activation_func = model.activation_func
        self.model_type = model.model_type
        self.activation_type = model.activation_type

        self.model = deepcopy(model)
        self.model.load_state_dict(torch.load(load_path, map_location="cpu"))
        self.model.eval()
        
    def generate_saliency(self, x, target):
        """vanilla gradient*input"""
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        o = self.model(x)
        grad_outputs = torch.zeros_like(o).scatter(1, target.unsqueeze(1), 1).detach()
        o.backward(gradient=grad_outputs)
        x.requires_grad_(requires_grad=False)
        
        return x.grad.clone() * x