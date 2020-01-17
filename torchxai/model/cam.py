__author__ = "simonjisu"

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import XaiModel, XaiHook

class GradCAM(XaiModel):
    """GradCAM"""
    def __init__(self, model, layers_name=None, norm_mode=1):
        """
        args:
        - layers_name
        - norm mode
            * 1 ( 0, 1) min-max normalization
            * 2 (-1, 1) min-max normalization
            * 3 mean-std normalization
        """
        super(GradCAM, self).__init__(model)
        self.norm_mode = norm_mode
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # DONE [RF] 0.1.2 features/layername
        # not to use `self._find_target_layer_idx` for resnet because is too hard to use
        if layers_name == None:
            # only supported to cnn models which have `convs` layer + ReLU activation
            target_idxes = self._find_target_layer_idx("convs", ["relu"])
            last_target_idx = target_idxes["relu"][-1]
            # get Rectified Conv Features Maps
            self.f_hook = XaiHook(self.model._modules["convs"][last_target_idx])
            self.b_hook = XaiHook(self.model._modules["convs"][last_target_idx])
        else:
            self.f_hooks = []
            self.b_hooks = []
            all_layers = self._find_target_layer(layers_name)
            self.num_hooks = len(all_layers)
            for l in all_layers:
                self.f_hooks.append(XaiHook(l))
                self.b_hooks.append(XaiHook(l))
        
        self._register_forward(self.f_hooks, hook_fn=None)
        self._register_backward(self.b_hooks, hook_fn=None)
    
    def cal_gradcam(self, key):
        # (B, C, H, W) > (B, C, 1, 1)
        alpha = self.global_avgpool(self.b_hooks[key].o[0])
        # sum( (B, C, 1, 1) * (B, C, H, W) , dim=1) > (B, 1, H, W)
        gradcam = torch.relu((alpha * self.f_hooks[key].o).sum(1, keepdim=True))
        return gradcam
        
    def post_processing(self, gradcam, H, W):
        """
        interpolate(up sample) & normalize
        https://pytorch.org/docs/stable/nn.functional.html#interpolate
        """
        gradcam = F.interpolate(gradcam, size=(H, W), mode="bilinear", align_corners=False)
        gradcam = self.normalization(gradcam)
        return gradcam
    
    def normalization(self, tensor):
        B, C, H, W = tensor.size()
        tensor = tensor.view(B, -1)
        t_min = tensor.min(dim=1, keepdim=True)[0]
        t_max = tensor.max(dim=1, keepdim=True)[0]
        t_mean = tensor.mean(dim=1, keepdim=True)
        t_std = tensor.std(dim=1, keepdim=True)
        if self.norm_mode == 1:
            tensor -= t_min
            tensor /= (t_max - t_min + 1e-10)
        elif self.norm_mode == 2:
            tensor -= t_min
            tensor *= 2
            tensor /= (t_max - t_min + 1e-10)
        elif self.norm_mode == 3:
            tensor -= t_mean
            tensor /= t_std
        return tensor.view(B, C, H, W)
        
    def get_attribution(self, x, targets, key=None):
        *_, H, W = x.size()
        if key is None:
            key = 0
        self.model.zero_grad()
        
        outputs = self.model(x)
        grad = self._one_hot(targets, module_name="fc")
        outputs.backward(grad)
        
        cams = self.cal_gradcam(key)
        cams = self.post_processing(cams, H, W)
        return cams.detach()


