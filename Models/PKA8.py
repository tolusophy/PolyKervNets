import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from Models.ModelPKN import Kerv2d
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim

class PKA8(nn.Module):
    def __init__(self, img_channels=3, num_classes=10):
        super(PKA8, self).__init__()
        self.kerv1 = Kerv2d(in_channels=img_channels, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.kerv2 = Kerv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.kerv3 = Kerv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.kerv4 = Kerv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.kerv5 = Kerv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.fc  = nn.Linear(in_features=1024, out_features= num_classes)
        self.drop = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):
        x = self.drop(self.kerv1(x))
        x = self.drop(self.kerv2(x))
        x = self.pool(x)
        x = self.drop(self.kerv3(x))
        x = self.drop(self.kerv4(x))
        x = self.drop(self.kerv5(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x