import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(Model, self).__init__()
        '''
        # Load the pretrained model from the Model Hub. 
        # Note that this was only made for VGG and ResNet models. 
        # You can add your pytorch models here too or create a whole model definition.
        '''
        
        if hasattr(models, model_name):
            self.model = getattr(models, model_name)(pretrained=pretrained)
        elif model_name == 'resnet10':
            self.model = getattr(models, 'resnet18')(pretrained=pretrained)
            self.model.layer3 = nn.Identity()
            self.model.layer4 = nn.Identity()
        elif model_name == 'resnet14':
            self.model = getattr(models, 'resnet18')(pretrained=pretrained)
            self.model.layer4 = nn.Identity()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        if 'resnet' in model_name:
            # Get the number of features in the last fully connected layer
            if model_name == 'resnet10':
                self.model.fc = nn.Linear(128, num_classes)
            if model_name == 'resnet14':
                self.model.fc = nn.Linear(256, num_classes)
            else:
                in_features = self.model.fc.in_features
                # Replace the last fully connected layer with a new linear layer
                self.model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Unsupported model type")

    def forward(self, x):
        x = self.model(x)
        return x