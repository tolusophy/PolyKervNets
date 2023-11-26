import argparse
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim
from Scripts.wrapper import *
from Scripts.model import *
from Scripts.pkn import *
from numpy import random
from torchsummary import summary
from momo import Momo, MomoAdam
torch.manual_seed(0)

torch.autograd.set_detect_anomaly(True)

color_jitter = transforms.ColorJitter(random.uniform(0.1, 0.5),random.uniform(0.1, 0.5),random.uniform(0.1, 0.5),random.uniform(0.01, 0.15))
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomApply([color_jitter], p=0.5),
                                transforms.RandomAutocontrast(p=0.5),
                                transforms.Resize(224),
                                transforms.transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(224),
                                transforms.transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=test_transform)

batch_size = 128

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

def train(model, train_loader, test_loader, loss_fn, optimizer, scheduler=None, num_epochs=100):
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in tqdm(enumerate(train_loader, 0), desc="Training", leave=True):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step(loss=loss)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        accuracy = 100 * correct_train / total_train
        train_accuracy.append(accuracy)
        train_loss.append(running_loss / len(train_loader))

        with torch.no_grad():
            rtest_loss = 0.0
            correct_test = 0
            total_test = 0

            for i, data in tqdm(enumerate(test_loader, 0), desc="Testing", leave=True):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                rtest_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        accuracy = 100 * correct_test / total_test
        test_accuracy.append(accuracy)

        # Calculate and store test loss
        test_loss.append(rtest_loss / len(test_loader))

        if scheduler is not None:
            scheduler.step(test_loss[-1])

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {train_loss[-1]:.4f} - Training Accuracy: {train_accuracy[-1]:.2f}% - Test Loss: {test_loss[-1]:.4f} - Test Accuracy: {test_accuracy[-1]:.2f}%")

    return model

def trainkd(teacher_model, student_model, train_loader, test_loader, teacher_loss_fn, student_loss_fn, teacher_optimizer, student_optimizer, student_scheduler=None, teacher_scheduler=None, num_epochs=100):
    teacher_train_loss = []
    student_train_loss_original_labels = []
    student_train_loss_teacher_outputs = []
    test_teacher_accuracy = []
    test_student_accuracy_original_labels = []
    test_student_accuracy_teacher_outputs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        teacher_running_loss = 0.0
        student_running_loss_original_labels = 0.0
        student_running_loss_teacher_outputs = 0.0
        teacher_correct_train = 0
        student_correct_train_original_labels = 0
        student_correct_train_teacher_outputs = 0
        total_train = 0

        for i, data in tqdm(enumerate(train_loader, 0), desc="Training", leave=True):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Teacher model training
            teacher_optimizer.zero_grad()
            teacher_outputs = teacher_model(inputs)
            teacher_loss = teacher_loss_fn(teacher_outputs, labels)
            teacher_loss.backward()
            teacher_optimizer.step(loss=teacher_loss)

            # Student model training with respect to teacher's outputs
            student_optimizer.zero_grad()
            teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)
            student_loss_teacher_outputs = student_loss_fn(nn.functional.log_softmax(student_outputs, dim=1), nn.functional.softmax(teacher_outputs / 5.0, dim=1))
            student_loss_teacher_outputs.backward()
            student_optimizer.step(loss=student_loss_teacher_outputs)
            student_outputs = student_model(inputs)
            student_loss_original_labels = teacher_loss_fn(student_outputs, labels)
            # student_loss_original_labels.backward()
            # student_optimizer.step(loss=student_loss_original_labels)

            teacher_running_loss += teacher_loss.item()
            student_running_loss_teacher_outputs += student_loss_teacher_outputs.item()
            student_running_loss_original_labels += student_loss_original_labels.item()
            
            _, teacher_predicted = torch.max(teacher_outputs.data, 1)
            _, student_predicted_original_labels = torch.max(student_outputs.data, 1)
            _, student_predicted_teacher_outputs = torch.max(student_outputs.data, 1)

            total_train += labels.size(0)
            teacher_correct_train += (teacher_predicted == labels).sum().item()
            student_correct_train_original_labels += (student_predicted_original_labels == labels).sum().item()
            student_correct_train_teacher_outputs += (student_predicted_teacher_outputs == labels).sum().item()

        teacher_accuracy = 100 * teacher_correct_train / total_train
        student_accuracy_original_labels = 100 * student_correct_train_original_labels / total_train
        student_accuracy_teacher_outputs = 100 * student_correct_train_teacher_outputs / total_train

        teacher_train_loss.append(teacher_running_loss / len(train_loader))
        student_train_loss_original_labels.append(student_running_loss_original_labels / len(train_loader))
        student_train_loss_teacher_outputs.append(student_running_loss_teacher_outputs / len(train_loader))
        test_teacher_accuracy.append(test_model(teacher_model, test_loader, device))
        test_student_accuracy_original_labels.append(test_model(student_model, test_loader, device))

        if student_scheduler is not None and teacher_scheduler is not None:
            student_scheduler.step()
            teacher_scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Teacher Training Loss: {teacher_train_loss[-1]:.4f} \
            - Teacher Test Accuracy: {test_teacher_accuracy[-1]:.2f}% \
            - Student Training Loss (Teacher Outputs): {student_train_loss_teacher_outputs[-1]:.4f} \
            - Student Training Loss (Original Labels): {student_train_loss_original_labels[-1]:.4f} \
            - Student Test Accuracy (Original Labels): {test_student_accuracy_original_labels[-1]:.2f}%")

    return student_model

def test_model(model, test_loader, device):
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader, 0), desc="Testing", leave=True):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    accuracy = 100 * correct_test / total_test
    model.train()
    return accuracy

torch.manual_seed(0)
model = Model(model_name='resnet50', num_classes=10, pretrained=True).to('cuda')
# model.load_state_dict(torch.load('model.pth'))

'''
you can either replace the convolution kernel or the activation. do not replace both!
actwrapper(model, ActPKN/ReactPKN) or convwrapper(model, PKN2d)
To change the kernel type in PKN2d, go to the pkn.py file and switch the kernel between RPKN and PKN. Default is RPKN
'''

actwrapper(model, ReactPKN)
# convwrapper(model, PKN2d) 

summary(model, (3,224,224))

optimizer = MomoAdam(model.parameters(), lr=5e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
loss_fn = nn.CrossEntropyLoss()

'use train() for normal training and trainkd() for knowledge distillation'

model = train(model, train_loader, test_loader, loss_fn, optimizer, scheduler, num_epochs=200)
torch.save(model.state_dict(), 'model.pth')
