__author__ = "simonjisu"

import torch
import torch.nn as nn
from .base import XaiModel, XaiHook

class relLinear(XaiHook):
    """relLinear"""
    def __init__(self, module, use_rho=False):
        """
        forward
        > input: (B, in_f)
        > output: (B, out_f)
        backward
        > lrp propagation with respect to previous input
        """
        super(relLinear, self).__init__(module)
        self.out_features = self.module.out_features
        self.use_rho = use_rho
        self.register_hook(backward=False, hook_fn=self.f_hook)
        self.register_hook(backward=True, hook_fn=self.b_hook)
    
    def f_hook(self, m, i, o):
        """
        forward hook
        i: (input,)
        o: output
        
        save forward input and output data
        """
        self.input = i[0].clone().data
        self.output = o.clone().data
    
    def b_hook(self, m, i, o):
        """
        backward hook
        i: (grad_bias, grad_input, grad_weight.T) -> backward output
        o: (gard_output,) -> backward input
        
        ### implementation method 1
        [Step 1]: (B, in_f, 1) * (1, in_f, out_f) = (B, in_f, out_f)
        [Step 2]: (B, 1, out_f), do not multiply `torch.sign(self.output.unsqueeze(1))` 
                  that returns `nan` in tensor
        [Step 3]: divide by s
        [Step 4]: (B, in_f, out_f) x (B, out_f, 1) = (B, in_f)
        
        ```
        # Step 1
        z = self.input.unsqueeze(-1) * w.transpose(0, 1).unsqueeze(0)
        # Step 2
        s = self.output.unsqueeze(1) + eps * torch.sign(self.output.unsqueeze(1))  
        # Step 3
        weight = z / s
        # Step 4
        r_next = torch.bmm(weight, r.unsqueeze(-1)).squeeze()
        ```
        ### implemetation method 2
        # Step 1: (B, out_f), do not multiply `torch.sign(self.output)` that returns `nan` in tensor
        # Step 2: (B, out_f) / (B, out_f) = (B, out_f)
        # Step 3: (B, in_f, out_f) * (B, out_f, 1) = (B, in_f)
        # Step 4: (B, in_f) x (B, in_f) = (B, in_f)
        
        ```
        # Step 1
        s = self.output + eps
        # Step 2
        e = r / s
        # Step 3
        c = torch.bmm(w.transpose(0, 1).expand(e.size(0), 
                                               self.module.in_features, 
                                               self.module.out_features), 
                      e.unsqueeze(-1)).squeeze(-1)
        # Step 4
        r_next = self.input * c
        ```
        """
        grad_bias, grad_in, grad_weight = i
        r = o[0]
        eps = 1e-6
        w = self.rho(self.module.weight)
        # Step 1
        s = self.output + eps
        # Step 2
        e = r / s
        # Step 3
        c = torch.bmm(w.transpose(0, 1).expand(e.size(0), 
                                               self.module.in_features, 
                                               self.module.out_features), 
                      e.unsqueeze(-1)).squeeze(-1)
        # Step 4
        r_next = self.input * c
        assert r_next.size(1) == self.module.in_features, "size of `r_next` is not correct"
        # for debugging
        self.r = r  
        self.r_next = r_next
        return (grad_bias, r_next, grad_weight)
        
    def rho(self, w):
        if self.use_rho:
            return torch.clamp(w, min=0)
        else:
            return w


class relReLU(XaiHook):
    """relReLU"""
    def __init__(self, module):
        super(relReLU, self).__init__(module)
        self.register_hook(backward=True, hook_fn=self.b_hook)
        
    def b_hook(self, m, i, o):
        """
        backward hook
        i: (input,) -> backward output
        o: (output,) -> backward input
        """
        r = o
        return r


class relConv2d(XaiHook):
    """relConv2d"""
    def __init__(self, module, use_rho=False):
        """
        forward
        > input: (B, C_in, H_in, W_in)
        > output: (B, C_out, H_out, W_out)
        backward
        > lrp propagation with respect to previous input
        """
        super(relConv2d, self).__init__(module)
        self.use_rho = use_rho
        self.register_hook(backward=False, hook_fn=self.f_hook)
        self.register_hook(backward=True, hook_fn=self.b_hook)
    
    def f_hook(self, m, i, o):
        """
        forward hook
        i: (input,)
        o: output
        
        save forward input and output data
        """
        self.input = i[0].clone().data
        self.output = o.clone().data
    
    def b_hook(self, m, i, o):
        """
        backward hook
        i: (grad_input, grad_weight, gard_bias) -> backward output
        o: (gard_output,) -> backward input
        
        ### implementation method 
        [Step 1]: (B, C_out, H_out, W_out), do not multiply `torch.sign(self.output)` 
                   that returns `nan` in tensor
        [Step 2]: (B, C_out, H_out, W_out) / (B, C_out, H_out, W_out) = (B, C_out, H_out, W_out)
        [Step 3]: (B, C_out, H_out, W_out) --> (B, C_in, H, W)
                  same as `self.gradprop(s*e)` or `(s*e).backward(); c=self.input.grad`
        [Stpe 4]: (B, C_in, H, W) x (B, C_in, H, W) = (B, C_in, H, W)
        
        ```
        # Step 1
        s = self.output + eps 
        # Step 2
        e = r / s
        # Step 3:
        c = self.gradprop(e, w)
        # Step 4
        r_next = self.input * c
        ```
        """
        _, grad_weight, grad_bias = i
        r = o[0]
        eps = 1e-6
        w = self.rho(self.module.weight)
        # Step 1
        s = self.output + eps 
        # Step 2
        e = r / s
        # Step 3:
        c = self.gradprop(e, w)
        # Step 4
        r_next = self.input * c

        # for debugging
        # self.r = r  
        # self.r_next = r_next
        return (r_next, grad_weight, grad_bias)
        
    def rho(self, w):
        if self.use_rho:
            return torch.clamp(w, min=0)
        else:
            return w

    def gradprop(self, x, w):
        """
        `ConvTransposed2d` can be seen as the gradient of `Conv2d` with respect to its input.
        """
        output_padding = self.cal_output_padding()
        c = torch.nn.functional.conv_transpose2d(x, 
                                                 weight=w, 
                                                 stride=self.module.stride, 
                                                 padding=self.module.padding, 
                                                 output_padding=output_padding)
        return c        

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
        S_h, S_w = self.module.stride
        K_h, K_w = self.module.kernel_size
        P_h, P_w = self.module.padding
        H_output_padding = H_in - ((H_out - 1)*S_h + K_h - 2*P_h)
        W_output_padding = W_in - ((W_out - 1)*S_w + K_w - 2*P_w)
        return (H_output_padding, W_output_padding)


class relMaxPool2d(XaiHook):
    """relMaxPool2d"""
    def __init__(self, module, use_rho=False):
        """
        forward
        > input: (B, C, H_in, W_in)
        > output: (B, C, H_out, W_out)
        backward
        > lrp propagation with respect to previous input
        """
        super(relMaxPool2d, self).__init__(module)
        self.use_rho = use_rho
        self.register_hook(backward=False, hook_fn=self.f_hook)
        self.register_hook(backward=True, hook_fn=self.b_hook)
    
    def f_hook(self, m, i, o):
        """
        forward hook
        i: (input,)
        o: output
        
        save forward input and output data
        """
        self.input = i[0].clone().data
        self.output = o.clone().data
        
    def b_hook(self, m, i, o):
        """
        backward hook
        i: (grad_input,) -> backward output
        o: (gard_output,) -> backward input
        
        ### implementation method 
        [Step 1]: (B, C, H_out, W_out), do not multiply `torch.sign(self.output)` 
                  that returns `nan` in tensor
        [Step 2]: (B, C, H_out, W_out) / (B, C, H_out, W_out) = (B, C, H_out, W_out)
        [Step 3]: (B, C, H_out, W_out) --> (B, C, H_in, W_in)
                  same as `self.gradprop(s*e)` or `(s*e).backward(); c=self.input.grad`
        [Stpe 4]: (B, C, H_in, W_in) x (B, C, H_in, W_in) = (B, C, H_in, W_in)
        
        ```
        # Step 1
        s = self.output + eps 
        # Step 2
        e = r / s
        # Step 3:
        c = self.gradprop(e)
        # Step 4
        r_next = self.input * c
        ```
        """        
        r = o[0]
        eps = 1e-6
        # Step 1
        s = self.output + eps
        # Step 2
        e = r / s
        # Step 3
        c = self.gradprop(e)
        # Step 4
        r_next = self.input * c
        
        # for debugging
        # self.r = r  
        # self.r_next = r_next
        return (r_next,)
    
    def gradprop(self, x):
        """
        get maxpooled switches first then unpool
        """
        _, switches = torch.nn.functional.max_pool2d(self.input, 
                                                     self.module.kernel_size, 
                                                     self.module.stride, 
                                                     self.module.padding, 
                                                     self.module.dilation, 
                                                     self.module.ceil_mode, 
                                                     return_indices=True)
        c = torch.nn.functional.max_unpool2d(x, switches, 
                                             self.module.kernel_size, 
                                             self.module.stride, 
                                             self.module.padding)
        return c


class LRP(XaiModel):
    """LRP"""
    def __init__(self, model, use_rho=False):
        """
        module_names: have to be sequential to forward network 
        """
        super(LRP, self).__init__(model)
        self.module_names = module_names
        self.use_rho = use_rho
        self.available_module = {
            nn.Linear: relLinear, 
            nn.Conv2d: relConv2d, 
            nn.MaxPool2d: relMaxPool2d, 
            nn.ReLU: relReLU
        }
        self.create_layers()
        
    def create_layers(self):
        for name, modules in self.model._modules.items():
            if isinstance(modules, nn.Sequential):
                for i, layer in enumerate(modules):
                    try:
                        modules[i] = self._create_layer(layer)
                    except KeyError as e:
                        print(f"{type(layer)} is not an available module.\nAvaiable:")
                        for k in self.available_module.keys():
                            print(f" - {k}")
            else:
                modules = self._create_layer(layer)      
    
    def forward(self, x):
        return self.model(x)
    
    def _create_layer(self, layer):
        if isinstance(layer, nn.ReLU):
            return self.available_module[type(layer)](layer)
        else:
            return self.available_module[type(layer)](layer, use_rho=self.use_rho)

    def get_attribution(self, x, targets):
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        output = self.forward(x)
        grad = self._one_hot(targets, module_name="fc")
        output.backward(grad)
        x_grad = x.grad.data.clone().detach()
        x.requires_grad_(requires_grad=False)
        return x_grad