# Module `SpatialConv2d` is modified from torch._ConvNd
# Reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py

__author__ = "simonjisu"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _pair

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, C, ratio):
        """
        Method in [arXiv:1807.06521]
        args:
        - C: channel of input features
        - ratio: reduction ratio

        returns:
        - Channel attention weight:Tensor (B, C, 1, 1)
        """
        super(ChannelAttention, self).__init__()
        assert isinstance(2*C // ratio, int), "`2*C // ratio` must be int "
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.shallow_net = nn.Sequential(
            nn.Linear(2*C, 2*C // ratio),
            nn.ReLU(),
            nn.Linear(2*C // ratio, 2*C),
        )
    
    def forward(self, x):
        # (B, C, H, W) > (B, 2*C, 1, 1)
        x = torch.cat([self.maxpool(x), self.avgpool(x)], dim=1)
        # (B, 2*C) > (B, 2*C//2) > (B, 2*C)
        x = self.shallow_net(x.squeeze(-1).squeeze(-1))
        # (B, C), (B, C)
        x_max, x_avg = torch.chunk(x, 2, dim=1)
        # not using softmax in paper: something like gate function
        x = torch.sigmoid(x_max + x_avg)
        return x.unsqueeze(-1).unsqueeze(-1)


class SpatialConv2d(nn.Module):
    """Spatial Conv2d Module"""
    __constants__ = ['stride', 'padding', 'bias', 'in_channels',
                     'out_channels', 'kernel_size']
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        """
        automatically calculate padding
        """
        super(SpatialConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride= stride
        self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def cal_padding_size(self, x, K, S):
        """not considering dilations & groups"""
        return int((S * (x-1) + K - x) / 2)
    
    def cal_sizes(self, x):
        B, C, H, W = x.size()
        P_H = self.cal_padding_size(H, self.kernel_size[0], self.stride[0])
        P_W = self.cal_padding_size(W, self.kernel_size[1], self.stride[1])
        self.padding = (P_H, P_W)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, x):
        self.cal_sizes(x)
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size, stride):
        """
        Method in [arXiv:1807.06521]
        args:
        - kernel_size
        - stride

        returns:
        - Spatial attention weight:Tensor (B, C, 1, 1)
        """
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        # same padding conv layer
        self.conv_layer = SpatialConv2d(2, 1, kernel_size, stride)

    def forward(self, x):
        # (B, C, H, W) > (B, 1, H, W)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_avg = torch.mean(x, dim=1, keepdim=True)
        # (B, 2, H, W)
        x = torch.cat([x_max, x_avg], dim=1)
        # (B, 2, H, W) > (B, 1, H, W)
        x = self.conv_layer(x)
        # return gated features
        return torch.sigmoid(x)


class CBAM(nn.Module):
    """Convolution Block Attention Module"""
    def __init__(self, C, ratio, kernel_size=7, stride=1):
        """
        Method in [arXiv:1807.06521]
        args:
        - C: channel of input features
        - ratio: reduction ratio
        - kernel_size
        - stride
         
        return:
         - attentioned features, size = (B, C, H, W)
        """
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(C, ratio)
        self.spatial_attn = SpatialAttention(_pair(kernel_size), _pair(stride))
        
    def forward(self, x, return_attn=False):
        """
        return: attentioned features, size = (B, C, H, W)
        """
        out = x
        c_attn = self.channel_attn(out)
        out = c_attn * out
        s_attn = self.spatial_attn(out)
        out = s_attn * out
        if return_attn:
            return out, (c_attn, s_attn)
        return out