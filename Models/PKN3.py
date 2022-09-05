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

class PKN3(nn.Sequential):
    def __init__(self, num_classes=10):
        super(PKN3, self).__init__()
        self.conv = Kerv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.kerv(x)
        x = x.reshape(-1, 256)
        x = self.fc(x)
        return x