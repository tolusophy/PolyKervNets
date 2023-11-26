import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d
import torch.nn.functional as F
import numpy as np

class ActPKN(torch.nn.Module):
    def __init__(self, cp=1.0, dp=2, dropout=False, learnable=True):
        super(ActPKN, self).__init__()
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=learnable))
        self.dp = dp
        self.dropout = dropout

    def forward(self, x):
        out = (x + self.cp) ** self.dp
        if self.dropout:
            out = torch.nn.Dropout()(out)
        return out

class ReactPKN(torch.nn.Module):
    def __init__(self, ap=0.009, bp=0.5, cp=0.47, dp=2, dropout=False, learnable=True):
        super(ReactPKN, self).__init__()
        self.ap = torch.nn.parameter.Parameter(torch.tensor(ap, requires_grad=learnable))
        self.bp = torch.nn.parameter.Parameter(torch.tensor(bp, requires_grad=learnable))
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=learnable))
        self.dp = dp
        self.dropout = dropout

    def forward(self, x):
        out = (self.ap * (x ** self.dp)) + (self.bp * x) + self.cp
        if self.dropout:
            out = torch.nn.Dropout()(out)
        return out

class LinearKernel(torch.nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()
    
    def forward(self, x_unf, w, b):
        t = x_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        if b is not None:
            return t + b
        return t
        
class PKN(LinearKernel):
    def __init__(self, cp=1.0, dp=2, learnable=True):
        super(PKN, self).__init__()
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=learnable))
        self.dp = dp

    def forward(self, x_unf, w, b):
        return (self.cp + super(PolynomialKernel, self).forward(x_unf, w, b))**self.dp

class RPKN(LinearKernel):
    def __init__(self, ap=0.009, cp=0.47, dp=2, learnable=True):
        super(RPKN, self).__init__()
        self.ap = torch.nn.parameter.Parameter(torch.tensor(ap, requires_grad=learnable))
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=learnable))
        self.dp = dp

    def forward(self, x_unf, w, b):
        return ((self.cp + super(PolynomialKernel, self).forward(x_unf, w, b))**self.dp)*self.ap

class PKN2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_fn=RPKN,
                 stride=1, padding=0, dilation=1, groups=1, bias=None,
                 padding_mode='zeros'):
        '''
        Follows the same API as torch Conv2d except kernel_fn.
        kernel_fn should be an instance of the above kernels.
        '''
        super(PKN2d, self).__init__(in_channels, out_channels, 
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