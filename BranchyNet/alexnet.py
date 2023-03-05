#!/usr/bin/env python3

from models.AlexNet import AlexNetWithExistsCIFAR10
from utils.functions import train_exit, train_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
import torchvision
import matplotlib

import os
import numpy as np
from datetime import datetime as dt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.ToTensor()

batch_size = 600

train_data   = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_data    = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = AlexNetWithExistsCIFAR10(exit_loss_weights=[20, 5, 1]).to(device)

train_exit(model, 2, train_loader=train_loader, test_loader=test_loader, device=device)
train_model(model, train_loader=train_loader, test_loader=test_loader, device=device)
train_exit(model, 1, train_loader=train_loader, test_loader=test_loader, device=device)
train_model(model, train_loader=train_loader, test_loader=test_loader, device=device)

