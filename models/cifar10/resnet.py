__author__ = "simonjisu"

from torchxai.base import XaiBase
import torch
import torch.nn as nn


class ResNetBlock(XaiBase):
    def __init__(self):
        super(ResNetBlock, self).__init__()