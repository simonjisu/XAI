import torch
import torch.nn as nn

class saliencyMNIST(nn.Module):
    def __init__(self, model):
        super(saliencyMNIST, self).__init__()
        """
        model_type: "dnn", "cnn"
        activation_type: "relu", "tanh", "sigmoid", "softplus"
        """
        super(saliencyMNIST, self).__init__()

        self.activation_func = model.activation_func
        self.model_type = model.model_type
        self.activation_type = model.activation_type
        self.convs_len = model.convs_len
        self.fc_len = model.fc_len
        # TODO: rewrite code
        self.model = deepcopy(model)
        self.model.eval()
        
    def generate_saliency(self, x, target):
        """vanilla gradient*input"""
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        output = self.model(x)
        grad_outputs = torch.zeros_like(output)
        grad_outputs[:, target] = 1

        output.backward(gradient=grad_outputs)
        x.requires_grad_(requires_grad=False)
        
        return x.grad.clone() * x