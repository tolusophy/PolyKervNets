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


cfg = {
    'PKV11': [[64, 'K'], 'A', [128, 'K'], 'A', [256, 'C'],
                [256, 'C'], 'A', [512, 'K'], [512, 'K'], 'A', 
                [512, 'C'], [512, 'C'] , 'A'],
    'PKV13': [[64, 'K'], [64, 'K'], 'A', [128, 'C'], [128, 'C'], 
                'A', [128, 'C'], [256, 'K'], 'A', [512, 'K'], 
                [512, 'C'], 'A', [512, 'C'], [512, 'C'], 'A'],
    'PKV16': [[64, 'K'], [64, 'K'], 'A', [128, 'C'], [128, 'C'], 
                'A', [256, 'C'], [256, 'K'], [256, 'K'], 'A', 
                [512, 'C'], [512, 'C'], [512, 'C'], 'A', [512, 'K'], 
                [512, 'C'], [512, 'C'], 'A'],
    'PKV19': [[64, 'K'], [64, 'K'], 'A', [128, 'C'], [128, 'C'], 'A', 
                [256, 'C'], [256, 'K'], [256, 'C'], [256, 'C'], 'A', 
                [512, 'C'], [512, 'K'], [512, 'C'], [512, 'C'], 'A', 
                [512, 'C'], [512, 'K'], [512, 'C'], [512, 'C'], 'A'],
    }


class PKV(nn.Module):
    def __init__(self, vgg_name):
        super(PKV, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                if x[1] == 'K':
                    layers += [Kerv2d(in_channels, x[0], kernel_size=3, padding=1),
                            nn.BatchNorm2d(x[0]),
                            nn.Dropout(p=0.2, inplace=False)]
                    in_channels = x[0]
                else:
                    layers += [nn.Conv2d(in_channels, x[0], kernel_size=3, padding=1),
                            nn.BatchNorm2d(x[0]),
                            nn.Dropout(p=0.2, inplace=False)]
                    in_channels = x[0]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)