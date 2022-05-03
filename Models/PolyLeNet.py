#loading the important libraries

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from Models.PolyKervNet import Kerv2d
import tenseal as ts
from CustomDataset import DatasetFromSubset
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim


#PolyLeNet
class PolyLeNet(nn.Sequential):
    def __init__(self,img_channels, num_classes):
        super(PolyLeNet, self).__init__()
        self.kerv1 = Kerv2d(img_channels, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.kerv2 = Kerv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(self.kerv1(x))
        x = self.pool(self.kerv2(x))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x