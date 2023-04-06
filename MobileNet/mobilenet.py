#!/usr/bin/env python3

import argparse
import sys
sys.path.append('..')

from models.MobileNet import *
from utils.functions import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
import torchvision
import matplotlib

import torch

import os
import numpy as np
from datetime import datetime as dt

parser = argparse.ArgumentParser(description='Train model - optionally load or save it')

parser.add_argument("--load", help="Filename to load from")
parser.add_argument("--save", help="Filename to save to")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.ToTensor()

batch_size = 600

train_data   = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_data    = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = MobileNet_v1(in_fts=3).to(device)

if args.load is not None:
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['model_state_dict'])    

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

epochs = 20
criterion_name = 'cross_entropy'

train_strategy = 'backbone_all'

if criterion_name == 'custom':
    criterion=CrossEntropyConfidence(device)
else:
    criterion_name = 'cross_entropy'
    criterion=nn.CrossEntropyLoss()

set_writer(f'runs/{train_strategy}_{criterion_name}_{dt_string}')

summary(model, (1, 3, 32, 32))
summary(model)

train_regular_model(model, train_loader=train_loader, test_loader=test_loader, device=device, epochs=epochs, criterion=criterion, lr=0.01)

if args.save is not None:
    save_dict = { 
        'model_state_dict': model.state_dict()
    }    
    torch.save(save_dict, args.save)


