#!/usr/bin/env python3

import argparse
import sys
sys.path.append('..')

from models.MawiNet import *
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = MawiNetWithExitsCIFAR10().to(device)

files = Path('../../datasets/balanced/2019/VIEGAS/01').iterdir()

df = pd.DataFrame()

for file in files:
    temp = pd.read_csv(file)
    df = pd.concat([df, temp])
    
df = df.drop(['MAWILAB_taxonomy', 'MAWILAB_distance', 'MAWILAB_nbDetectors', 'MAWILAB_label'], axis=1)

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

data = []
for val in X_train.values:
    val = np.append(val, 0).reshape(7,7)
    data.append(val)
    
X_train = torch.FloatTensor(data).view(len(data), 1, 7, 7).to(device)
y_train = y_train.values

data = []
for val in X_test.values:
    val = np.append(val, 0).reshape(7,7)
    data.append(val)
    
X_test = torch.FloatTensor(data).view(len(data), 1, 7, 7).to(device)
y_test = y_test.values

train_data = []
for i in range(len(X_train)):
    train_data.append((X_train[i], y_train[i]))

test_data = []
for i in range(len(X_test)):
    train_data.append((X_test[i], y_test[i]))

summary(model, (1, 1, 7, 7))
summary(model)
print(model)

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

set_writer(f'runs/{model.__class__.__name__}_{dt_string}')

epochs = 20
batch_size = 1000

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

train_model(model, train_loader=train_loader, test_loader=test_loader, device=device, epochs=epochs)

