import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim
from Scripts.model import model

torch.autograd.set_detect_anomaly(True)

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ColorJitter(brightness=.5, hue=.3),
                                transforms.Resize(256),
                                transforms.RandomCrop(224),
                                transforms.transforms.ToTensor(), 
                                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])

test_transform = transforms.Compose([transforms.Resize(256),
                                transforms.RandomCrop(224),
                                transforms.transforms.ToTensor(), 
                                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=test_transform)

batch_size = 32

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

def main(model, train_loader, test_loader, n_epochs):
    # model in training mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=3e-4)
    clr = optim.lr_scheduler.CyclicLR(optimizer, base_lr=3e-7, max_lr=3e-4)
    steplr = optim.lr_scheduler.StepLR(optimizer,step_size=50, gamma=0.7)
    model.train()
    for epoch in range(1, n_epochs+1):
        train_accuracy = 0
        train_samples = 0
        train_loss = 0.0
        for batch_idx,(data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(output, dim=-1)
            train_samples += predictions.size(0)
            train_accuracy += (predictions == targets).sum()
            train_loss += loss.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader)
        clr.step()
        steplr.step()

        with torch.no_grad():
            model.eval()
            test_loss = 0
            test_accuracy = 0
            test_samples = 0
            for batch_idx,(data, targets) in enumerate(tqdm(test_loader)):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                scores = model(data)
                loss = criterion(scores,targets)
                predictions = torch.argmax(scores, dim=-1)
                test_accuracy += (predictions == targets).sum()
                test_samples += predictions.size(0)
                test_loss += loss.item() 
            t_a = (test_accuracy / test_samples)*100
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss), f"  Test Accuracy: {t_a:.3f}")
            
    return model

PolyKerv = main(model, train_loader, test_loader, 1000)