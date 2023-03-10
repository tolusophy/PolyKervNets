##Define your model here i.e.
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d
import torch.nn.functional as F
import numpy as np
#you can use PKNs as either convolution or activation incase your model is pretrained
#and you want to remove the ReLU functions
from .wrapper import polykerv2d, pkn_act 
from .resnet import ResNet
from .pkn import Kerv2d
from .kernel import PolynomialKernel as PK
from torchsummary import summary

#Here are 3 examples of how to use PKNs

#Example 1

# model = ResNet.from_name('resnet10_2')
# model.fc = (model.fc.in_features, 10)
# polykerv2d(model)

#Example 2

# model = ResNet.from_pretrained('resnet18', num_classes=10) 
# #only 18, 32, 50, 101 and 152 are pretrained
# pkn_act(model)

#Example 3

nn.Kerv2d = Kerv2d
class model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        self.kerv = PK()

        self.conv1 = nn.Kerv2d(3, 64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Kerv2d(64, 256, kernel_size=7, stride=2)
        self.bn = nn.BatchNorm2d(256)

        self.pool = nn.AvgPool2d(2)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=7, stride=2)
        self.fc1 = nn.Linear(30976, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn(self.conv2(x))
        x = self.drop(x)
        
        x = self.bn(self.conv3(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.bn3(self.fc1(x))
        x = self.drop(self.kerv(x))

        x = self.fc(x)
        return x

model = model(num_classes=10)

summary(model, (3,224,224))