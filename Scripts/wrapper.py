import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d
import torch.nn.functional as F
import numpy as np
from .pkn import *

def actwrapper(module, act_fn=None):
    for name, layer in module._modules.items():
        actwrapper(layer, act_fn)
        
        if act_fn is not None and isinstance(layer, nn.ReLU):
            # Replace ReLU layer with the specified activation function
            act = act_fn()
            module._modules[name] = act
        
        if act_fn is not None and isinstance(layer, nn.MaxPool2d):
            # Create replacement AvgPool2D layer with the same kernel size and stride
            avg_pool = nn.AvgPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
            )
            module._modules[name] = avg_pool

def convwrapper(module, kernel=None):
    for name, layer in module._modules.items():
        convwrapper(layer, kernel)
        
        if kernel is not None and isinstance(layer, nn.Conv2d):
            pkn = PKN2d(layer.in_channels, layer.out_channels, layer.kernel_size, kernel,
                layer.stride, layer.padding, layer.dilation, layer.groups, bias,
                layer.padding_mode)
            module._modules[name] = pkn
        
        if kernel is not None and isinstance(layer, nn.ReLU):
            # Replace ReLU layer with the specified activation function
            act = nn.Identity()
            module._modules[name] = act
        
        if kernel is not None and isinstance(layer, nn.MaxPool2d):
            # Create replacement AvgPool2D layer with the same kernel size and stride
            avg_pool = nn.AvgPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
            )
            module._modules[name] = avg_pool