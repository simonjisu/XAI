# Simple Version of ResNet in PyTorch
# modified some codes from the reference
# Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
__author__ = "simonjisu"
__all__ = [
    "BasicBlock", "CBAMBlock", "ResNetSimple", "ResNetMnist", "ResNetMnistCBAM", "ResNetCifar10", "ResNetCifar10CBAM"
]

import torch
import torch.nn as nn
from torchxai.base import XaiBase, XaiHook
import torchxai.module as xaimodule
from collections import OrderedDict

def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)


# --- Blocks ---
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU()
        self.relu_last = nn.ReLU()  # Need to record for attribution method
        self.conv2 = conv1x1(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        # first Conv Block
        # (B, C_in, H, W) > (B, C_out, H, W)
        # downsample: (B, C_in, H, W) > (B, C_out, (H+1)/2, (W+1)/2)
        o = self.conv1(x)  
        o = self.bn1(o)
        o = self.relu(o)
        
        # second Conv Block
        # (B, C_out, H, W) > (B, C_out, H, W)
        # downsample (B, C_out, (H+1)/2, (W+1)/2) > (B, C_out, (H+1)/2, (W+1)/2)
        o = self.conv2(o)  
        o = self.bn2(o)
        
        if self.downsample is not None:
            # Conv1x1(C_in, C_out, stride=2)
            # BatchNorm2d(C_out)
            # (B, C_in, H, W) > (B, C_out, (H+1)/2, (W+1)/2)
            identity = self.downsample(x)  
        
        o += identity
        o = self.relu_last(o)
        return o


class CBAMBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None, **kwargs):
        super(CBAMBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.cbam1 = xaimodule.CBAM(C=out_channels, ratio=16, kernel_size=7, stride=1)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU()
        self.relu_last = nn.ReLU()  # Need to record for attribution method
        self.conv2 = conv1x1(out_channels, out_channels)
        self.cbam2 = xaimodule.CBAM(C=out_channels, ratio=16, kernel_size=7, stride=1)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        # first Conv Block
        # (B, C_in, H, W) > (B, C_out, H, W)
        # downsample: (B, C_in, H, W) > (B, C_out, (H+1)/2, (W+1)/2)
        o = self.conv1(x)
        o = self.cbam1(o)
        o = self.bn1(o)
        o = self.relu(o)
        
        # second Conv Block
        # (B, C_out, H, W) > (B, C_out, H, W)
        # downsample (B, C_out, (H+1)/2, (W+1)/2) > (B, C_out, (H+1)/2, (W+1)/2)
        o = self.conv2(o)
        o = self.cbam2(o)
        o = self.bn2(o)
        
        if self.downsample is not None:
            # Conv1x1(C_in, C_out, stride=2)
            # BatchNorm2d(C_out)
            # (B, C_in, H, W) > (B, C_out, (H+1)/2, (W+1)/2)
            identity = self.downsample(x)  
        
        o += identity
        o = self.relu_last(o)
        return o


# ---Models---
class ResNetSimple(XaiBase):
    """ResNetSimple"""
    def __init__(self, block, layers, img_c=3, num_classes=1000, 
                zero_init_residual=False, 
                norm_layer=None):
        """
        Simple Version of ResNet in PyTorch
        modified some codes from the reference
        Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        """
        super(ResNetSimple, self).__init__()
        # --- Creating Layer Part ---
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.in_c = 64
        self.out_c_dict = {k: v for k, v in enumerate([64, 128, 256, 512])}

        self.conv1 = nn.Conv2d(img_c, self.in_c, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_c)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.resnet_layers = self.make_layer(block, layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.out_c_dict[self.last_layer_idx] * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) or isinstance(m, CBAMBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def make_layer(self, block, layers):
        """
        create resnet layer
        """
        resnet_layers = []
        for i in range(len(layers)):
            stride = 1 if i == 0 else 2
            # if use `setattr`
            # the model contains residual blocks like `self.layer1`, `self.layer2` ...
            # you can check the id is same, `id(model.layer1) == id(model.resnet_layers[0])`
            # easier to run forward return the sequential blocks 
            # --- code ---
            setattr(
                self,
                f"layer{i+1}", 
                self._make_layer(block, out_c=self.out_c_dict[i], blocks=layers[i], stride=stride)
            )
            resnet_layers.append(self._modules[f"layer{i+1}"])
            # ------------
            # alternative can use like below, but can't use `self.layer1` style
            # for convenient transfer learning 
            # --- code ---
            # resnet_layers.append(self._make_layer(
            #     block, 
            #     out_c=self.out_c_dict[i], 
            #     blocks=layers[i], 
            #     stride=stride))
            # ------------
        self.last_layer_idx = i
        return nn.Sequential(*resnet_layers)
    
    def _make_layer(self, block, out_c, blocks, stride=1):
        """create resnet blocks"""
        norm_layer = self._norm_layer
        downsample = None
        # downsample module 
        if stride != 1 or self.in_c != out_c * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_c, out_c * block.expansion, stride),
                norm_layer(out_c * block.expansion),
            )

        layers = []
        # downsample if exists
        layers.append(block(self.in_c, out_c, stride, downsample, norm_layer))
        self.in_c = out_c * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_c, out_c, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.resnet_layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def register_attention_hooks(self):
        self.attn_hooks = OrderedDict()
        for name, m in self.named_modules():
            if isinstance(m, xaimodule.CBAM):
                # each CBAMBlock contains 2 CBAM module
                # if layers = [2, 2, 2], total CBAM module will be (2*2)*3
                c_attn_hook = XaiHook(m.channel_attn)
                s_attn_hook = XaiHook(m.spatial_attn)
                self.attn_hooks[f"{name}-c_attn"] = c_attn_hook
                self.attn_hooks[f"{name}-s_attn"] = s_attn_hook
        self._register_forward(list(self.attn_hooks.values()))

    def close(self):
        """close all hooks"""
        self._reset_hooks(list(self.attn_hooks.values()))

    def forward_map(self, x):
        self.register_attention_hooks()
        self._reset_maps()
        o = self.forward(x)
        for k, hook in self.attn_hooks.items():
            self._save_maps(k, hook.o)
        return o
        
# create exists resnet
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetSimple(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def ResNetMnist(pretrained=False, progress=True, **kwargs):
    """resnetmnist model"""
    return _resnet('resnetmnist', BasicBlock, [2, 2, 2], pretrained, progress,
                   img_c=1, num_classes=10, **kwargs)

def ResNetMnistCBAM(pretrained=False, progress=True, **kwargs):
    """resnetmnist model with CBAM"""
    return _resnet('resnetmnist', BasicBlock, [2, 2, 2], pretrained, progress,
                   img_c=1, num_classes=10, **kwargs)

def ResNetCifar10(pretrained=False, progress=True, **kwargs):
    """resnetmnist model"""
    return _resnet('resnetmnist', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   img_c=3, num_classes=10, **kwargs)

def ResNetCifar10CBAM(pretrained=False, progress=True, **kwargs):
    """resnetmnist model with CBAM"""
    return _resnet('resnetmnist', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   img_c=3, num_classes=10, **kwargs)