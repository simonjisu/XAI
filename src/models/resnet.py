# Simple Version of ResNet in PyTorch
# modified some codes from the reference
# Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
__author__ = "simonjisu"
__all__ = [
    "BasicBlock", "BasicBlockCBAM", "Bottleneck", "BottleneckCBAM", "ResNetBase", 
    "ResNetMnist", "ResNetMnistCBAM", "ResNetMnistANR", "ResNetCifar10", "ResNetCifar10CBAM", "ResNetCifar10ANR"
]

import torch
import torch.nn as nn
from torchxai.base import XaiBase, XaiHook
from torchxai.module import cbam, anr
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
            # Conv1x1(C_in, C_out*expansion(1), stride=2)
            # BatchNorm2d(C_out*expansion(1))
            # (B, C_in, H, W) > (B, C_out, (H+1)/2, (W+1)/2)
            identity = self.downsample(x)  
        
        o += identity
        o = self.relu_last(o)
        return o


class BasicBlockCBAM(BasicBlock):
    expansion = 1
    __constants__ = ['downsample']
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(BasicBlockCBAM, self).__init__(in_channels, out_channels, stride, downsample, norm_layer)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.cbam1 = cbam.CBAM(C=out_channels, ratio=16, kernel_size=7, stride=1)
        self.cbam2 = cbam.CBAM(C=out_channels, ratio=16, kernel_size=7, stride=1)

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
            # Conv1x1(C_in, C_out*expansion(1), stride=2)
            # BatchNorm2d(C_out*expansion(1))
            # (B, C_in, H, W) > (B, C_out, (H+1)/2, (W+1)/2)
            identity = self.downsample(x)  
        
        o += identity
        o = self.relu_last(o)
        return o


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(out_channels * (64 / 64.)) / 1
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride)
        self.bn2 = norm_layer(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.relu_last = nn.ReLU()
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
        o = self.relu(o)
        # third Conv Block
        # (B, C_out, H, W) > (B, C_out*expansion(4), H, W)
        # downsample (B, C_out, (H+1)/2, (W+1)/2) > (B, C_out*expansion(4), (H+1)/2, (W+1)/2)
        o = self.conv3(o)
        o = self.bn3(o)

        if self.downsample is not None:
            # Conv1x1(C_in, C_out*expansion(4), stride=2)
            # BatchNorm2d(C_out*expansion(4))
            # (B, C_in, H, W) > (B, C_out*expansion(4), (H+1)/2, (W+1)/2)
            identity = self.downsample(x)

        o += identity
        o = self.relu_last(o)

        return o


class BottleneckCBAM(Bottleneck):
    expansion = 4
    __constants__ = ['downsample']
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(BottleneckCBAM, self).__init__(in_channels, out_channels, stride, downsample, norm_layer)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.cbam1 = cbam.CBAM(C=out_channels, ratio=16, kernel_size=7, stride=1)
        self.cbam2 = cbam.CBAM(C=out_channels, ratio=16, kernel_size=7, stride=1)
        self.cbam3 = cbam.CBAM(C=out_channels * self.expansion, ratio=16, kernel_size=7, stride=1)

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
        o = self.relu(o)
        # third Conv Block
        # (B, C_out, H, W) > (B, C_out*expansion(4), H, W)
        # downsample (B, C_out, (H+1)/2, (W+1)/2) > (B, C_out*expansion(4), (H+1)/2, (W+1)/2)
        o = self.conv3(o)
        o = self.cbam3(o)
        o = self.bn3(o)

        if self.downsample is not None:
            # Conv1x1(C_in, C_out*expansion(4), stride=2)
            # BatchNorm2d(C_out*expansion(4))
            # (B, C_in, H, W) > (B, C_out*expansion(4), (H+1)/2, (W+1)/2)
            identity = self.downsample(x)

        o += identity
        o = self.relu_last(o)

        return o



# --- Models ---

class ResNetBase(XaiBase):
    """ResNetBase"""
    def __init__(self, block, layers, img_c, num_cls, zero_init_residual, norm_layer=None, build_layer=True):
        """
        Simple Version of ResNet in PyTorch
        modified some codes from the reference
        Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        """
        super(ResNetBase, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_cls = num_cls 
        self.in_c = 64
        self.out_c_dict = {k: v for k, v in enumerate([64, 128, 256, 512])}
        self.zero_init_residual = zero_init_residual
        
        self.conv1 = nn.Conv2d(img_c, self.in_c, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_c)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.out_c_dict[len(layers)-1] * block.expansion, num_cls)
        if build_layer:
            # build resnet block at init
            self.resnet_layers = self.make_layer(block, layers)
            self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
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
            # ------------
            resnet_layers.append(self._modules[f"layer{i+1}"])
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

    def register_activation_hooks(self):
        self.hooks = OrderedDict()
        for name, m in self.named_modules():
            if isinstance(m, nn.ReLU) and "relu_last" in name:
                hook = XaiHook(m)
                self.hooks[f"{name}"] = hook
        self._register_forward(list(self.hooks.values()))

    def forward_map(self, x):
        self.register_activation_hooks()
        self._reset_maps()
        o = self.forward(x)
        for k, hook in self.hooks.items():
            self._save_maps(k, hook.o)
        return o


class ResNetCBAM(ResNetBase):
    """ResNetCBAM"""
    def __init__(self, block, layers, img_c=3, num_cls=1000, zero_init_residual=False, norm_layer=None):
        super(ResNetCBAM, self).__init__(block, layers, img_c, num_cls, zero_init_residual, norm_layer, build_layer=True)

    def register_attention_hooks(self):
        self.hooks = OrderedDict()
        for name, m in self.named_modules():
            if isinstance(m, cbam.CBAM):
                # each CBAMBlock contains 2 CBAM module
                # if layers = [2, 2, 2], total CBAM module will be (2*2)*3
                c_attn_hook = XaiHook(m.channel_attn)
                s_attn_hook = XaiHook(m.spatial_attn)
                attentioned_hook = XaiHook(m)
                self.hooks[f"{name}-c_attn"] = c_attn_hook
                self.hooks[f"{name}-s_attn"] = s_attn_hook
                self.hooks[f"{name}-output"] = attentioned_hook
        self._register_forward(list(self.hooks.values()))

    def close(self):
        """close all hooks"""
        self._reset_hooks(list(self.hooks.values()))

    def forward_map(self, x):
        self.register_attention_hooks()
        self._reset_maps()
        o = self.forward(x)
        for k, hook in self.hooks.items():
            self._save_maps(k, hook.o)
        return o


class ResNetANR(ResNetBase):
    """ResNetANR"""
    def __init__(self, block, layers, img_c=3, num_cls=1000, zero_init_residual=False, norm_layer=None, 
            n_head=4, reg_weight=0.0, gate_fn="softmax"):
        super(ResNetANR, self).__init__(block, layers, img_c, num_cls, zero_init_residual, norm_layer, build_layer=False)
        self.n_head = n_head
        self.reg_weight = reg_weight
        self.resnet_layers = self.make_layer(block, layers)

        self.global_attn_gate = anr.GlobalAttentionGate(
            self.out_c_dict[len(layers)-1]*block.expansion, len(layers), gate_fn=gate_fn)
        self.init_weight()

    def make_layer(self, block, layers):
        resnet_layers = []
        for i in range(len(layers)):
            stride = 1 if i == 0 else 2
            # Block Layer
            setattr(
                self,
                f"layer{i+1}", 
                self._make_layer(block, out_c=self.out_c_dict[i], blocks=layers[i], stride=stride)
            )
            resnet_layers.append(self._modules[f"layer{i+1}"])
            if i+1 != len(layers):
                # Attention Layer
                setattr(
                    self,
                    f"attn{i+1}",
                    anr.AttentionModule(self.out_c_dict[i]*block.expansion, self.n_head, self.num_cls, reg_weight=self.reg_weight)
                )
                resnet_layers.append(self._modules[f"attn{i+1}"])
        return nn.ModuleList(resnet_layers)

    def _forward_impl(self, x):
        reg_loss = 0.0
        hypothesis = []
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for layer in self.resnet_layers:
            if isinstance(layer, anr.AttentionModule):
                hypo = layer(x)
                reg_loss += layer.reg_loss()
                hypothesis.append(hypo.unsqueeze(1))  # + (B, 1, L)
            else:
                x = layer(x)

        last_conv = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        hypothesis.append(x.unsqueeze(1))  # + (B, 1, L)

        o = self.global_attn_gate(last_conv, hypothesis)
        self.reg_loss = reg_loss
        return o

    def forward(self, x):
        return self._forward_impl(x)

    def register_attention_hooks(self):
        self.hooks = OrderedDict()
        for name, m in self.named_modules():
            if isinstance(m, anr.AttentionModule):
                attn_hook = XaiHook(m.attn_heads)
                self.hooks[f"{name}-attn"] = attn_hook
        self._register_forward(list(self.hooks.values()))

    def close(self):
        """close all hooks"""
        self._reset_hooks(list(self.hooks.values()))

    def forward_map(self, x):
        self.register_attention_hooks()
        self._reset_maps()
        o = self.forward(x)
        for k, hook in self.hooks.items():
            self._save_maps(k, hook.o)
        return o


# --- create resnet ---

def _resnet(arch, block, layers, **kwargs):
    model = arch(block, layers, **kwargs)
    return model

# Mnist

def ResNetMnist():
    """mnist model"""
    return _resnet(ResNetBase, BasicBlock, [2, 2], img_c=1, num_cls=10, zero_init_residual=False)

def ResNetMnistCBAM():
    """mnist model with CBAM"""
    return _resnet(ResNetCBAM, BasicBlockCBAM, [2, 2], img_c=1, num_cls=10, zero_init_residual=False)

def ResNetMnistANR():
    """mnist model with ANR"""
    return _resnet(ResNetANR, BasicBlock, [2, 2], img_c=1, num_cls=10, zero_init_residual=False, 
        norm_layer=None, n_head=4, reg_weight=0.01, gate_fn="softmax")

# Cifar10

# def ResNetCifar10():
#     """cifar10 model"""
#     return _resnet(ResNetBase, BasicBlock, [2, 2, 2, 2], img_c=3, num_cls=10, zero_init_residual=False)

# def ResNetCifar10CBAM():
#     """cifar10 model with CBAM"""
#     return _resnet(ResNetCBAM, BasicBlockCBAM, [2, 2, 2, 2], img_c=3, num_cls=10, zero_init_residual=False)

# def ResNetCifar10ANR():
#     """cifar10 model with ANR"""
#     return _resnet(ResNetANR, BasicBlock, [2, 2, 2, 2], img_c=3, num_cls=10, zero_init_residual=False, 
#         norm_layer=None, n_head=4, reg_weight=0.01, gate_fn="softmax")

def ResNetCifar10():
    """cifar10 model"""
    return _resnet(ResNetBase, BasicBlock, [2, 2, 2], img_c=3, num_cls=10, zero_init_residual=True)

def ResNetCifar10CBAM():
    """cifar10 model with CBAM"""
    return _resnet(ResNetCBAM, BasicBlockCBAM, [2, 2, 2], img_c=3, num_cls=10, zero_init_residual=True)

def ResNetCifar10ANR():
    """cifar10 model with ANR"""
    return _resnet(ResNetANR, BasicBlock, [2, 2, 2], img_c=3, num_cls=10, zero_init_residual=True, 
        norm_layer=None, n_head=4, reg_weight=0.01, gate_fn="softmax")
        