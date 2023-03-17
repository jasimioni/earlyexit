#!/usr/bin/env python3

import argparse

from models.AlexNet import AlexNetWithExistsCIFAR10
from utils.functions import *

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

model = AlexNetWithExistsCIFAR10(exit_loss_weights=[1, 0.5, 0.2]).to(device)

if args.load is not None:
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['model_state_dict'])    


# Pre treinar backbone
#train_exit(model, 2, train_loader=train_loader, test_loader=test_loader, device=device, 
#           backbone_parameters='path', epochs=5, criterion=CrossEntropyConfidence(device))

'''
# Opcao de treinar cada seção individualmente

train_exit(model, 0, train_loader=train_loader, test_loader=test_loader, device=device, backbone_parameters='section', epochs=5)
show_exits_stats(model, test_loader, device=device)
train_exit(model, 1, train_loader=train_loader, test_loader=test_loader, device=device, backbone_parameters='section', epochs=5)
show_exits_stats(model, test_loader, device=device)
train_exit(model, 2, train_loader=train_loader, test_loader=test_loader, device=device, backbone_parameters='section', epochs=5)
show_exits_stats(model, test_loader, device=device)
'''

# Treinar todas as saidas juntas
train_model(model, train_loader=train_loader, test_loader=test_loader, device=device, epochs=5, criterion=CrossEntropyConfidence(device))

# Treinar só a saída 0
# train_exit(model, 0, train_loader=train_loader, test_loader=test_loader, device=device, backbone_parameters='none', epochs=5)

# Treinar só a saída 1
# train_exit(model, 1, train_loader=train_loader, test_loader=test_loader, device=device, backbone_parameters='none', epochs=5)

# Treinar só a saída 2
# train_exit(model, 2, train_loader=train_loader, test_loader=test_loader, device=device, backbone_parameters='none', epochs=5)

# Treinar só a saída 0
# train_exit(model, 0, train_loader=train_loader, test_loader=test_loader, device=device, backbone_parameters='section', epochs=5)

# Treinar só a saída 1
# train_exit(model, 1, train_loader=train_loader, test_loader=test_loader, device=device, backbone_parameters='section', epochs=5)

# Treinar só a saída 2
# train_exit(model, 2, train_loader=train_loader, test_loader=test_loader, device=device, backbone_parameters='section', epochs=5)


show_exits_stats(model, test_loader, device=device)

for i, (X, y) in enumerate(test_data):
    X = X.to(device)
    x = X.view(1,3,32,32)
    print(f"Correct: {y} - {model.exits_certainty(x)}")

if args.save is not None:
    save_dict = { 
        'model_state_dict': model.state_dict()
    }    
    torch.save(save_dict, args.save)


# train_model(model, train_loader=train_loader, test_loader=test_loader, device=device, epochs=5)

