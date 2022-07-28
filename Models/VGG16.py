import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from PolyKervNet.PolyKervNet import Kerv2d
# import tenseal as ts
from CustomDataset import DatasetFromSubset
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim


class PolyVGG16(nn.Module):
    def __init__(self):
        super(PolyVGG16, self).__init__()
        self.conv1_1 = Kerv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = Kerv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = Kerv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = Kerv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = Kerv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = Kerv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = Kerv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = Kerv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = Kerv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = Kerv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = Kerv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = Kerv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = Kerv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.maxpool(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.maxpool(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x