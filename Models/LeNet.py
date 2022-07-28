#loading the important libraries

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from PolyKervNet.PolyKervNet import Kerv2d
# import tenseal as ts
from Scripts.CustomDataset import DatasetFromSubset
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim

class PolyLeNet(nn.Sequential):
    def __init__(self,img_channels=3, num_classes=10):
        super(PolyLeNet, self).__init__()
        self.kerv1 = Kerv2d(img_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.kerv2 = Kerv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, num_classes)

    def forward(self, x):
        x = self.pool(self.kerv1(x))
        x = self.pool(self.kerv2(x))
        x = x.reshape(-1, 16 * 5 * 5)
        x = self.fc1(x)
        return x