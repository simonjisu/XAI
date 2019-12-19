__author__ = "simonjisu"

import torch
import torch.nn as nn
from collections import OrderedDict


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, C, H, W, ratio):
        """
        Method in [arXiv:1807.06521]
        args:
         - C: channel of input features
         - H: height of input features
         - W: width of input features
         - hid_size: hidden size of shallow network
         - ratio: reduction ratio
        """
        super(ChannelAttention, self).__init__()
        kernel_size = (H, W)
        self.maxpool = nn.MaxPool2d(kernel_size)
        self.avgpool = nn.AvgPool2d(kernel_size)
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


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, H, W, K_H=7, K_W=7, S_H=1, S_W=1):
        """
        Method in [arXiv:1807.06521]
        args:
         - H: height of input features
         - W: width of input features
         - K_H: height of kernel size
         - K_W: width of kernel size
         - S_H: stride height of conv layer
         - S_W: stride width of conv layer
        """
        super(SpatialAttention, self).__init__()
        P_H = self.cal_padding_size(H, K_H, S_H)
        P_W = self.cal_padding_size(W, K_W, S_W)
        kernel_size = (K_H, K_W)
        stride = (S_H, S_W)
        padding = (P_H, P_W)
        # same padding conv layer
        self.conv_layer = nn.Conv2d(2, 1, kernel_size, stride, padding)
    
    def cal_padding_size(self, x, K, S):
        return int((S * (x-1) + K - x) / 2)
    
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
    def __init__(self, C, H, W, ratio, K_H=7, K_W=7, S_H=1, S_W=1):
        """
        Method in [arXiv:1807.06521]
        args:
         - C: channel of input features
         - H: height of input features
         - W: width of input features
         - ratio: reduction ratio
         - K_H: height of kernel size
         - K_W: width of kernel size
         - S_H: stride height of conv layer
         - S_W: stride width of conv layer
         
        return:
         - attentioned features, size = (B, C, H, W)
        """
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(C, H, W, ratio)
        self.spatial_attn = SpatialAttention(H, W, K_H, K_W, S_H, S_W)
        
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