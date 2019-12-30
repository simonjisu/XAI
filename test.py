import torch
import torch.nn as nn
from copy import deepcopy
from collections import defaultdict, OrderedDict

class XaiHook(nn.Module):
    def __init__(self, module):
        super(XaiHook, self).__init__()
        """
        Hook Handler Module
        
        supported register `module` hooks
        - Activations
        - Linear
        - Convd
        
        like RNN have to use `register_hook` to `torch.nn.Parameter` directly
        
        * Ref: https://pytorch.org/docs/master/nn.html#torch.nn.Module.register_backward_hook
        [Warnings]
        The current implementation will not have the presented behavior 
        for complex Module that perform many operations. In some failure cases, 
        `grad_input` and `grad_output` will only contain the gradients for a subset
        of the inputs and outputs. For such `Module`, you should use 
        `torch.Tensor.register_hook()` directly on a specific input or 
        output to get the required gradients.
        
        """
        self.module = module
    
    def zero_grad(self):
        self.module.zero_grad()

    def register_hook(self, backward=False, hook_fn=None):
        """
        defalut hook_function is save (module, input, output) to (m, i, o)
        if you want to use hook function, change `hook_function` 
        if `hook_function` returns `None` then the original input or output 
        will be flow into next / previous layer, but you can return a modifed
        output/gradient to change the original output/gradient.
        for a Conv2d layer example
        - forward: a `Tensor` type output
        - backward: (gradient_input, weight, bias)
        
        """
        def default_hook_fn(m, i, o):
            """
            forward
             - m: module class
             - i: forward input from previous layer
             - o: forward output to next layer
            backward
             - m: module class
             - i: gradient input to next layer (backward out)
             - o: gradient output from previous layer (backward in)

            args:
             * i, o: tuple type
            """
            self.m = m
            self.i = i
            self.o = o
            
        if hook_fn is None:
            self.hook_fn = default_hook_fn
        else:
            self.hook_fn = hook_fn
        if not backward:
            self.hook = self.module.register_forward_hook(self.hook_fn)
        else:
            self.hook = self.module.register_backward_hook(self.hook_fn)
            
    def close(self):
        self.hook.remove()


class XaiBase(nn.Module):
    def __init__(self):
        super(XaiBase, self).__init__()
        """
        - need to define XaiHook class to use
        - defalut hook_function is save (module, input, output) to (m, i, o)
          if you want to use hook function, change `hook_function` 
        """
        self._reset_maps()
    
    def _reset_maps(self):
        self.maps = OrderedDict()
        
    def _save_maps(self, layer_name, x):
        self.maps[layer_name] = x    
        
    def _register(self, hooks, backward=False, hook_fn=None):
        """
        - need to define XaiHook class to use
        - defalut hook_function is save (module, input, output) to (m, i, o)
          if you want to use hook function, change `hook_function` 
        """
        if not isinstance(hooks, list):
            hooks = [hooks]
        for hook in hooks:
            hook.register_hook(backward=backward, hook_fn=hook_fn)
    
    def _register_forward(self, hooks, hook_fn=None):
        self._register(hooks, backward=False, hook_fn=hook_fn)
        
    def _register_backward(self, hooks, hook_fn=None):
        self._register(hooks, backward=True, hook_fn=hook_fn)
    
    def _reset_hooks(self, hooks):
        if not isinstance(hooks, list):
            hooks = [hooks]
        for hook in hooks:
            hook.close()

    def _return_indices(self, layers, on=True):
        """
        support for cnn layer which have `nn.MaxPool2d`,
        you can turn on/off pooling indices.
        please define a forward function to use it in your model
        '''
        # in your model
        def forward_switch(self, x):
            switches = OrderedDict()
            self.return_indices(on=True)
            for idx, layer in enumerate(self.convs):
                if isinstance(layer, nn.MaxPool2d):
                    x, indices = layer(x)
                    switches[idx] = indices
                else:
                    x = layer(x)
            self.return_indices(on=False)
            return x, switches
        '''
        """
        if on:
            for layer in layers:
                if isinstance(layer, nn.MaxPool2d):
                    layer.return_indices = True
        else:
            for layer in layers:
                if isinstance(layer, nn.MaxPool2d):
                    layer.return_indices = False  
                    
                    
class XaiModel(XaiBase):
    def __init__(self, model):
        super(XaiModel, self).__init__()
        self.model = deepcopy(model)
        self.model.cpu()
        self.model.eval()
        
    def _one_hot(self, targets, module_name):
        """
        one hot vectorize the target tensor for classification purpose.
        the `module` with respect to `module_name` must have `out_features` attribution.
        args:
        - targets: torch.LongTensor, target classes that have size of mini-batch
        - module_name: str, feature name for Fully-Connected Network or any Task-specific Network
        return:
        - one hot vector of targets
        """
        assert isinstance(targets, torch.LongTensor), "`targets` must be `torch.LongTensor` type"
        assert isinstance(module_name, str), "`module_name` must be `str` type"
        modules = self.model._modules[module_name]
        if isinstance(modules, nn.Sequential):
            last_layer = modules[-1]
        else:
            last_layer = modules
        try:
            last_layer.out_features 
        except AttributeError as e:
            is_linear = isinstance(last_layer, nn.Linear)
            print(f"last layer of module `{module_name}` doesn't have `out_features` attribute")
            print()
            if not is_linear:
                print(f"type of the last layer is `{type(last_layer)}`")
                print("the last layer is not `torch.nn.linear.Linear` class")
                print("create `.out_featrues` attribution in the custom module")
                
        target_size = last_layer.out_features
        B = targets.size(0)
        one_hot = torch.zeros((B, target_size))
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        return one_hot.to(targets.device)
    
    def _find_target_layer_idx(self, module_name, layer_names):
        assert isinstance(layer_names, list) or isinstance(layer_names, tuple), "use list for `layer_names`"
        layer_names = [l.lower() for l in layer_names]
        idxes = defaultdict(list)
        modules = self.model._modules[module_name]
        assert isinstance(modules, nn.Sequential), "use this function for `nn.Sequential` type modules"
        for idx, layer in modules.named_children():
            l_name = type(layer).__name__.lower()
            if l_name in layer_names:
                idxes[l_name].append(int(idx))

        return idxes
    
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
        
    def __call__(self, x):
        return self.module(x)
    
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
        w = self.rho(self.module.weight).data
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
        # self.r = r  
        # self.r_next = r_next
        return (grad_bias, r_next, grad_weight)
#         return r_next
        
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
        
    def __call__(self, x):
        return self.module(x)
    
    def rho(self, w):
        if self.use_rho:
            return torch.clamp(w, min=0)
        else:
            return w
        
    def b_hook(self, m, i, o):
        """
        backward hook
        i: (input,) -> backward output
        o: (output,) -> backward input
        """
        r = o
        return (r,)
#         return r

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
        
    def __call__(self, x):
        return self.module(x)
    
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
#         return r_next
        
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
            
    def __call__(self, x):
        return self.module(x)
    
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
#         return r_next
    
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
    def __init__(self, model, module_names, use_rho=False):
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
#         modules = self.model._modules[module_name]
        
#         self.convs = 
        for module_name in self.module_names:
            modules = self.model._modules[module_name]
            layers = []
            if isinstance(modules, nn.Sequential):
                for i, layer in enumerate(modules):
                    try:
                        layers.append(self._create_layer(layer))
                    except KeyError as e:
                        print(f"{type(layer)} is not an available module.\nAvaiable:")
                        for k in self.available_module.keys():
                            print(f" - {k}")
                layers = nn.Sequential(*layers)
            else:
                layers = self._create_layer(layer)
            self.__setattr__(module_name, layers)        
        
    def _create_layer(self, layer):
        if isinstance(layer, nn.ReLU):
            return self.available_module[type(layer)](layer)
        else:
            return self.available_module[type(layer)](layer, use_rho=self.use_rho)

    def forward(self, x):        
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def get_attribution(self, x, targets):
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        output = self.forward(x)
        grad = self._one_hot(targets, module_name="fc")
        output.backward(grad)
        x_grad = x.grad.data.clone()
        x.requires_grad_(requires_grad=False)
        return x_grad

class Cnn(XaiBase):
    def __init__(self):
        super(Cnn, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # (B, 1, 28, 28) > (B, 32, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 24, 24) > (B, 32, 12, 12)
            nn.Conv2d(32, 64, 3),  # (B, 32, 12, 12) > (B, 64, 10, 10)
            nn.ReLU(), 
            nn.MaxPool2d(2),  # (B, 64, 10, 10) > (B, 64, 5, 5)
            nn.Conv2d(64, 128, 2),  # (B, 128, 5, 5) > (B, 128, 4, 4)
            nn.ReLU(), 
            nn.MaxPool2d(2),  # (B, 128, 4, 4) > (B, 128, 2, 2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128*2*2, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):        
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward_map(self, x):
        self._reset_maps()
        for i, layer in enumerate(self.convs):
            layer_name = type(layer).__name__.lower()
            if layer_name == "relu":
                x, attns = layer(x, return_attn=True)
                self._save_maps(f"{i}"+layer_name, attns)
            else:
                x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = Cnn()
    x = torch.randn(1, 1, 28, 28)
    targets = torch.LongTensor([2])
    module_names = ["convs", "fc"]
    lrp = LRP(model, module_names)
    x_lrp = lrp.get_attribution(x, targets)
    plt.imshow(x_lrp.squeeze())
    plt.show()