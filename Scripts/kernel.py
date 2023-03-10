import numpy as np
import torch
import torch.nn.functional as F

class PolynomialKernel(torch.nn.Module):
    def __init__(self, c=0.5, degree=2):
        super(PolynomialKernel, self).__init__()
        self.c = torch.nn.parameter.Parameter(torch.tensor(c), requires_grad=False)
        self.degree = torch.nn.parameter.Parameter(torch.tensor(degree), requires_grad=False)

    def forward(self, x):
        out = (x + self.c) ** self.degree
        out = torch.nn.Dropout()(out)
        return out