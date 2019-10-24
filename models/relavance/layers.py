import torch
import torch.nn as nn

class relLinear(nn.Linear):
    def __init__(self, linear):
        super(nn.Linear, self).__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight  # (out_f, in_f)
        self.bias = linear.bias  # out_f
        self.input = None
        self.output = None
        self.register()
        
    def register(self):
        self.register_forward_hook(self.hook_function)
    
    def hook_function(self, *x):
        _, i, o = x
        self.input = i[0].data
        self.output = o.data
        
    def rho(self, w, use_rho):
        if use_rho:
            return torch.clamp(w, min=0)
        else:
            return w
        
    def relprop(self, r, use_rho=False):
        """
        lrp method
            > * must run after `self.forward`
            > 
            > forward shape
            > input: (B, *, in_f)
            > output: (B, *, out_f)

        - relprop shape
        r = (l+1)-th layer: (B, *, out_f)
        r_next = l-th layer: (B, *, in_f)
        
        if rho==True:
        function rho(w) is applied
        """
        eps = 1e-6
        ### implementation method 1
        ## Step 1: (B, in_f, 1) * (1, in_f, out_f) = (B, in_f, out_f)
        # z = self.input.unsqueeze(-1) * self.rho(self.weight).transpose(0, 1).unsqueeze(0)
        ## Step 2: (B, 1, out_f)
        # s = self.output.unsqueeze(1) + eps * torch.sign(self.output.unsqueeze(1))
        ## Step 3: (B, 1, out_f)
        # weight = z / s
        ## Step 4: (B, in_f, out_f) x (B, out_f, 1) = (B, in_f)
        # r_next = torch.bmm(weight, r.unsqueeze(-1)).squeeze()
        
        ### implemetation method 2
        # Step 1: (B, out_f) 
        s = self.output + eps * torch.sign(self.output)  
        # Step 2: (B, out_f) / (B, out_f) = (B, out_f)
        e = r / s
        # Step 3: (B, in_f, out_f) * (B, out_f, 1) = (B, in_f)
        c = torch.bmm(self.weight.transpose(0, 1).expand(e.size(0), self.in_features, self.out_features), 
                      e.unsqueeze(-1)).squeeze()
        # Step 4: (B, in_f) x (B, in_f) = (B, in_f)
        r_next = self.input * c
        
        assert r_next.size(1) == self.in_features, "size of `r_next` is not correct"
        return r_next
    
class relReLU(nn.ReLU): 
    def relprop(self, r): 
        return r