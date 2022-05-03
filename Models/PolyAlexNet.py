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


#PolyAlexNet
class PolyAlexNet(nn.Module):
    def __init__(self, img_channels,num_classes):
        super(PolyAlexNet, self).__init__()
        self.kerv1 = Kerv2d(in_channels=img_channels, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)
        self.kerv2 = Kerv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.kerv3 = Kerv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.kerv4 = Kerv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.kerv5 = Kerv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 9216, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=num_classes)

    def forward(self, x):
        x = self.kerv1(x)
        x = self.pool(x)
        x = self.kerv2(x)
        x = self.pool(x)
        x = self.kerv3(x)
        x = self.kerv4(x)
        x = self.kerv5(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x