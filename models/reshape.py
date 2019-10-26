import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self):
        """
        reshape layer
        - forward: flatten at convs > linear
        - backward: unflatten at linear > convs     
        """
        super(Reshape, self).__init__()
    
    def __str__(self):
        return "Reshape()"
    
    def forward(self, x, backward=False):
        """
        reshape input to output
        input: (B, C, H, W)
        output: (B, C*H*W)
        
        if backward:
        must run after `self.forward`
        input: (B, C*H*W)
        output: (B, C, H, W)
        """
        if backward:
            return x.view(-1, *self.sizes)
        else:
            B, *self.sizes = x.size()
            return x.view(B, -1)
        
    def relprop(self, x, use_rho=False):
        return self.forward(x, backward=True)