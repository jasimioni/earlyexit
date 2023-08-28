#!/usr/bin/env python3

import argparse
import sys
sys.path.append('..')

from models.TryNets import *
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

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from pathlib import Path

if len(sys.argv) == 1:
    print("Please provide a save name")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_data   = CustomMawiDataset(year='2016', month='01', as_matrix=True)
test_data    = CustomMawiDataset(year='2016', month='02', as_matrix=True)

model = TryNetWithExits().to(device)

summary(model, (1, 1, 7, 7))
summary(model)
print(model)

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

filename = f'{model.__class__.__name__}_{sys.argv[1]}_{dt_string}'

set_writer(f'runs/{filename}')

epochs = 20
batch_size = 1000

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()

# train_model(model, train_loader=train_loader, test_loader=test_loader, device=device, epochs=epochs)
train_exit(model, 0, backbone_parameters='section', train_loader=train_loader, test_loader=test_loader, device=device, epochs=epochs)
train_exit(model, 1, backbone_parameters='section', train_loader=train_loader, test_loader=test_loader, device=device, epochs=epochs)
train_exit(model, 2, backbone_parameters='section', train_loader=train_loader, test_loader=test_loader, device=device, epochs=epochs)

save_dict = { 
    'model_state_dict': model.state_dict()
}    

torch.save(save_dict, f'saves/{filename}') 