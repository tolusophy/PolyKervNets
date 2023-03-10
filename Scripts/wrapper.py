import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d
import torch.nn.functional as F
import numpy as np
from .pkn import Kerv2d
from .kernel import PolynomialKernel as PK

nn.Kerv2d = Kerv2d
def polykerv2d(module):
    for name, layer in module._modules.items():
        polykerv2d(layer)
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            # Create replacement layer
            bias = layer.bias is not None
            pkn = nn.PKN2d(
                layer.in_channels, layer.out_channels, layer.kernel_size, kernel,
                layer.stride, layer.padding, layer.dilation, layer.groups, bias,
                layer.padding_mode
            )

            module._modules[name] = pkn

def pkn_act(module):
    for name, layer in module._modules.items():
        pkn_act(layer)
        if isinstance(layer, torch.nn.modules.activation.ReLU):
            # Create replacement layer
            pkn = PK()

            module._modules[name] = pkn