#!/usr/bin/env python3

import argparse
import sys
sys.path.append('..')

from models.MooreNet import *
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

train_data   = CustomMawiDataset(author='MOORE', year='2016', month='01', as_matrix=True)

model = MooreNetWithExits().to(device)

summary(model, (1, 1, 8, 8))
summary(model)
print(model)

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f'{model.__class__.__name__}_{sys.argv[1]}_{dt_string}'

set_writer(f'runs/{filename}')

epochs = 5
batch_size = 5000

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()

train_model(model, train_loader=train_loader, test_loader=train_loader, device=device, epochs=epochs)

data   = CustomMawiDataset(author='MOORE', year='2016', month='01', as_matrix=True)
loader = DataLoader(data, batch_size=batch_size, shuffle=False)

for year in (2016, 2017, 2018, 2019):
    year = str(year)

    for month in range(12):
        month = f'{month+1:02d}'

        # data = CustomMawiDataset(author='MOORE', year=year, month=month, as_matrix=True)
        # loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            total = 0
            correct = [0, 0, 0]
            for i, (X, y) in enumerate(loader):
                X = X.to(device)
                y = y.to(device)

                results = model(X)
                
                c_acc = [ 0, 0, 0 ]

                for exit, result in enumerate(results):
                    predicted = torch.max(result.data, 1)[1]

                    c_acc[exit] += (predicted == y).sum()
                    
                    # c_results.append([cnf, predicted])
                            
                    correct[exit] += (predicted == y).sum()

                acc = ' | '.join([ f'{100*corr/len(y):.4}% ' for corr in c_acc ])

                print(f'Results: {acc}')

                total += len(y)

            print(f"{year}-{month}")
            print(f'Accuracy: {100*correct[0]/total:.2f} | {100*correct[1]/total:.2f} | {100*correct[2]/total:.2f}')

            accuracies = ' | '.join([ f'{100*acc/total:.6}% ' for acc in correct ])

            print(f'Accuracies per exit (for all dataset): {accuracies}')

            sys.exit(1)