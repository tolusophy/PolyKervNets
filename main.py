import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
# import tenseal as ts
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim
from Scripts.Train_Test import train_test as tt
from Models.OneCNN import Poly1Net
from Models.LeNet import PolyLeNet
from Models.AlexNet import PolyAlexNet
from Models.VGG16 import PolyVGG16
from Models.Resnet18 import PolyResNet18
from Models.Resnet18_3 import PolyResNet18_3
from Models.Resnet18_2 import PolyResNet18_2



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

batch_size = 4

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

PolyKerv = tt(PolyResNet18_2(), train_loader, test_loader, 200)