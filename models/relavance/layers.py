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
        w = self.rho(self.weight, use_rho)
        ### implementation method 1
        ## (B, in_f, 1) * (1, in_f, out_f) = (B, in_f, out_f)
        #z = self.input.unsqueeze(-1) * self.rho(self.weight).transpose(0, 1).unsqueeze(0)
        ## (B, 1, out_f)
        #s = self.output.unsqueeze(1) + eps * torch.sign(self.output.unsqueeze(1))  
        #weight = z / s
        ## (B, in_f, out_f) x (B, out_f, 1) = (B, in_f)
        #r_next = torch.bmm(weight, r.unsqueeze(-1)).squeeze()
        
        ### implemetation method 2
        # Step 1: (B, out_f) 
        s = self.output + eps * torch.sign(self.output)  
        # Step 2: (B, out_f) / (B, out_f) = (B, out_f)
        e = r / s
        # Step 3: (B, in_f, out_f) * (B, out_f, 1) = (B, in_f)
        c = torch.bmm(w.transpose(0, 1).expand(e.size(0), self.in_features, self.out_features), 
                      e.unsqueeze(-1)).squeeze(-1)
        # Step 4: (B, in_f) x (B, in_f) = (B, in_f)
        r_next = self.input * c
        
        assert r_next.size(1) == self.in_features, "size of `r_next` is not correct"
        return r_next
    
class relConv2d(nn.Conv2d):
    def __init__(self, conv2d):
        super(nn.Conv2d, self).__init__(conv2d.in_channels, 
                                        conv2d.out_channels, 
                                        conv2d.kernel_size, 
                                        conv2d.stride, 
                                        conv2d.padding,
                                        conv2d.dilation,
                                        conv2d.transposed,
                                        conv2d.output_padding,
                                        conv2d.groups, 
                                        None,  # init of bias
                                        conv2d.padding_mode)
        self.weight = conv2d.weight
        self.bias = conv2d.bias
        
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
    
    def cal_output_padding(self):
        """
        calculate output_padding size
        - size of height or width: (X_in + 2P - K) / S + 1 = X_out
        - output_padding = X_in - ((X_out - 1) * S + K - 2P)

        * what is output_padding?
        from PyTorch Document:
        https://pytorch.org/docs/stable/nn.html#convtranspose2d

        The padding argument effectively adds `dilation * (kernel_size - 1) - padding` amount of zero padding to 
        both sizes of the input. This is set so that when a `Conv2d` and a `ConvTranspose2d` are initialized with 
        same parameters, they are inverses of each other in regard to the input and output shapes. 
        However, when `stride > 1`, `Conv2d` maps multiple input shapes to the same output shape. 
        `output_padding` is provided to resolve this ambiguity by effectively increasing 
        the calculated output shape on one side. Note that output_padding is only used to find output shape, 
        but does not actually add zero-padding to output.
        """
        H_in, W_in = self.input.size()[2:]
        H_out, W_out = self.output.size()[2:]
        S_in, S_out = self.stride
        K_in, K_out = self.kernel_size
        P_in, P_out = self.padding
        H_output_padding = H_in - ((H_out - 1)*S_in + K_in - 2*P_in)
        W_output_padding = W_in - ((W_out - 1)*S_out + K_out - 2*P_out)
        return (H_output_padding, W_output_padding)
    
    def gradprop(self, x):
        """
        `ConvTransposed2d` can be seen as the gradient of `Conv2d` with respect to its input.
        """
        output_padding = self.cal_output_padding()
        c = torch.nn.functional.conv_transpose2d(x, weight=self.weight, stride=self.stride, 
                                                 padding=self.padding, output_padding=output_padding)
        return c
    
    def relprop(self, r, use_rho=False):
        """
        lrp method
            > * must run after `self.forward`
            > 
            > forward shape
            > input: (B, C_in, H, W)
            > output: (B, C_out, H_out, W_out)

        - relprop shape
        r = (l+1)-th layer: (B, C_out, H_out, W_out)
        r_next = l-th layer: (B, C_in, H, W)
        
        if rho==True:
        function rho(w) is applied
        """
        eps = 1e-6
        w = self.rho(self.weight, use_rho)
        # Step 1: (B, C_out, H_out, W_out) 
        s = self.output + eps * torch.sign(self.output)  
        # Step 2: (B, C_out, H_out, W_out) / (B, C_out, H_out, W_out) = (B, C_out, H_out, W_out)
        e = r / s
        # Step 3: (B, C_out, H_out, W_out) --> (B, C_in, H, W)
        # same as `self.gradprop(s*e)` or `(s*e).backward(); c=self.input.grad`
        c = self.gradprop(e)
        # Step 4: (B, C_in, H, W) x (B, C_in, H, W) = (B, C_in, H, W)
        r_next = self.input * c
        return r_next
    
class Reshape(nn.Module):
    def __init__(self):
        """
        reshape layer
        - forward: flatten at convs > linear
        - backward: unflatten at linear > convs     
        """
        super(Reshape, self).__init__()
    
    def forward(self, x):
        """
        reshape input to output
        input: (B, C, H, W)
        output: (B, C*H*W)
        """
        self.B, self.C, self.H, self.W = x.size()
        return x.view(B, -1)
    
    def relprop(self, x):
        """
        lrp method
            > * must run after `self.forward`
            > 
            > forward shape
            > input: (B, C, H, W)
            > output: (B, C*H*W)

        - relprop shape
        r = (l+1)-th layer: (B, C*H*W)
        r_next = l-th layer: (B, C, H, W)
        """
        return x.view(-1, self.C, self.H, self.W)