#!/usr/bin/env python3

import argparse
import sys
sys.path.append('..')

from models.AlexNet import *
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

train_data   = CustomMawiDataset(year='2019', month='01', as_matrix=False)
test_data    = CustomMawiDataset(year='2019', month='01', as_matrix=False)

model = AlexNetMawi().to(device)

summary(model, (1, 48))
summary(model)
print(model)

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

set_writer(f'runs/{model.__class__.__name__}_{dt_string}')

epochs = 5
batch_size = 100

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()

train_regular_model(model, train_loader=train_loader, test_loader=test_loader, device=device, epochs=epochs)
