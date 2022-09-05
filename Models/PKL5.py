import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from ModelPKN import Kerv2d
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim

class PKL5(nn.Sequential):
    def __init__(self,img_channels=3, num_classes=10):
        super(PKL5, self).__init__()
        self.kerv1 = Kerv2d(img_channels, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.kerv2 = Kerv2d(6, 16, 5)
        self.fc = nn.Linear(16 * 5 * 5, num_classes)

    def forward(self, x):
        x = self.pool(self.kerv1(x))
        x = self.pool(self.kerv2(x))
        x = x.reshape(-1, 16 * 5 * 5)
        x = self.fc(x)
        return x