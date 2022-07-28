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

#Poly1Net
class Poly1Net(nn.Sequential):
    def __init__(self, num_classes=10):
        super(Poly1Net, self).__init__()
        self.kerv1 = Kerv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.kerv1(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        return x