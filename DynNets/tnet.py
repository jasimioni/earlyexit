#!/usr/bin/env python3

import argparse
import sys
sys.path.append('..')

from models.DynSizeNet import *
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
import re

import torch

import os
import numpy as np
from datetime import datetime as dt

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from pathlib import Path

save = sys.argv[1]
print(save)
m = re.search('^([^_]+)_([^_]+)_([^_]+)_([^_]+)_([\d_]+)_2023', save)
m_type, author, year, month, convs = m.groups()

convs = [ int(i) for i in re.split('_', convs) ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

as_matrix = m_type == 'matrix'

sample_data = CustomMawiDataset(author=author, year=year, month='01', as_matrix=as_matrix)

if m_type == 'matrix':
    model = DynNetGen(input_sample=sample_data, conv_filters=convs).to(device)

checkpoint = torch.load(f'saves/{save}')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

batch_size = 5000

for year in (2016, 2017, 2018, 2019):
    year = str(year)

    for month in range(12):
        month = f'{month+1:02d}'

        data = CustomMawiDataset(author=author, year=year, month=month, as_matrix=as_matrix)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            total = 0
            correct = 0
            for i, (X, y) in enumerate(loader):
                X = X.to(device)
                y = y.to(device)

                result = model(X)
                
                predicted = torch.max(result.data, 1)[1]

                correct += (predicted == y).sum()
                total += len(y)
                    
            print(f'{year}-{month}: {100*correct/total:.2f}%')
