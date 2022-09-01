#Importing the required libraries

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d
import torch.nn.functional as F
import numpy as np

class PolynomialKernel(torch.nn.Module):
    def __init__(self, c=1.0, degree=2):
        super(PolynomialKernel, self).__init__()
        self.c = torch.nn.parameter.Parameter(torch.tensor(c), requires_grad=False)
        self.degree = torch.nn.parameter.Parameter(torch.tensor(degree), requires_grad=False)

    def forward(self, x, w):
        w = w.view(w.size(0), -1).t()
        out = (x.matmul(w) + self.c) ** self.degree
        return out
        

class Kerv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_fn=PolynomialKernel,
                 stride=1, padding=0, dilation=1, groups=1, bias=None,
                 padding_mode='zeros'):
        '''
        Follows the same API as torch Conv2d except kernel_fn.
        kernel_fn should be an instance of the above kernels.
        '''
        super(Kerv2d, self).__init__(in_channels, out_channels, 
                                           kernel_size, stride, padding,
                                           dilation, groups, bias, padding_mode)
        self.kernel_fn = kernel_fn()
   
    def compute_shape(self, x):
        h = (x.shape[2] + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w = (x.shape[3] + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return h, w
    
    def forward(self, x):
        x_unf = torch.nn.functional.unfold(x, self.kernel_size, self.dilation,self.padding, self.stride)
        h, w = self.compute_shape(x)
        return self.kernel_fn(x_unf, self.weight, self.bias).view(x.shape[0], -1, h, w)