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

#Defining the Kervolution Class

class Kerv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True,
            kernel_type='polynomial', learnable_kernel=False, kernel_regularizer=False,
            balance=0, power=2, gamma=1):

        super(Kerv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.kernel_type = kernel_type
        self.learnable_kernel, self.kernel_regularizer = learnable_kernel, kernel_regularizer
        self.balance, self.power, self.gamma = balance, power, gamma

        # parameter for kernel type
        if learnable_kernel == True:
            self.balance = nn.Parameter(torch.cuda.FloatTensor([balance] * out_channels), requires_grad=True).view(-1, 1)
            self.gamma   = nn.Parameter(torch.cuda.FloatTensor([gamma]   * out_channels), requires_grad=True).view(-1, 1)


    def forward(self, input):
        minibatch, in_channels, input_width, input_hight = input.size()
        assert(in_channels == self.in_channels)
        input_unfold = F.unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        input_unfold = input_unfold.view(minibatch, 1, self.kernel_size[0]*self.kernel_size[1]*self.in_channels, -1)
        weight_flat  = self.weight.view(self.out_channels, -1, 1)
        output_width = (input_width - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        output_hight = (input_hight - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

        if self.kernel_type == 'linear':
            output = (input_unfold * weight_flat).sum(dim=2)

        elif self.kernel_type == 'polynomial':
            output = ((input_unfold * weight_flat).sum(dim=2) + self.balance)**self.power

        else:
            raise NotImplementedError(self.kernel_type+' kervolution not implemented')

        if self.bias is not None:
            output += self.bias.view(self.out_channels, -1)

        return output.view(minibatch, self.out_channels, output_width, output_hight)



if __name__ == '__main__':
    kerv = Kerv2d(in_channels=2,              # input height
                  out_channels=3,             # n_filters
                  kernel_size=3,              # filter size
                  stride=1,                   # filter movement/step
                  padding=1,                  # input padding
                  kernel_type='polynomial',   # kernel type
                  learnable_kernel=True)      # enable learning parameters

    n_batch, in_channels, n_feature = 5, 2, 5
    x = torch.FloatTensor(n_batch, in_channels, n_feature).random_().cuda()
    kerv1d = Kerv1d(in_channels=in_channels, out_channels=2, kernel_size=3, kernel_type='polynomial', learnable_kernel=True).cuda()
    y = kerv1d(x)
    print(x.shape, y.shape)