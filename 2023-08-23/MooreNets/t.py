#!/usr/bin/env -S python3 -u

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchinfo import summary
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import re

from datetime import datetime
import os
import time
import sys
sys.path.append('..')

from utils.functions import *
from models.MooreNet import *

os.environ['PYTHONUNBUFFERED'] = "1"

if len(sys.argv) == 1:
    print("Please provide a file to load")
    sys.exit(1)

loadfile = sys.argv[1]
dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

batch_size = 5000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = MooreNetWithExits().to(device)

summary(model, (1, 1, 8, 8))
summary(model)
print(model)

model.load_state_dict(torch.load(f'saves/{loadfile}'))
# model.eval()
# model.set_measurement_mode()
# Run one inference to "load" everything
# with torch.no_grad():
#    model(torch.rand(1, 1, 8, 8).to(device))

thresholds = [ 0.5, 0.5, 0 ]

seq = 0
for year in (2016, 2017, 2018, 2019):
    year = str(year)

    for month in range(12):
        month = f'{month+1:02d}'

        data = CustomMawiDataset(author='MOORE', year=year, month=month, as_matrix=True)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        print(f'{year}-{month}')

        with torch.no_grad():
            total = 0
            correct = 0
            correct = [0, 0, 0]
            correct_exit = [0, 0, 0]
            chosen_exit  = [0, 0, 0]
            progression = {}
            for i, (X, y) in enumerate(loader):
                X = X.to(device)
                y = y.to(device)

                results = model(X)

                # print(X)
                # print(results)

                # sys.exit(1)

                c_results = []
                            
                for exit, result in enumerate(results):
                    cnf, predicted = torch.max(nn.functional.softmax(result, dim=-1), 1)

                    print(f'Exit: {exit}, Acc: {(predicted == y).sum()/len(y) * 100:.2f}%')
                    
                    c_results.append([cnf, predicted])
                            
                    correct[exit] += (predicted == y).sum()
                """    
                for i in range(len(y)):
                    for exit, res in enumerate(c_results):
                        if res[0][i] > thresholds[exit]:
                            chosen_exit[exit] += 1
                            if res[1][i] == y[i]:
                                correct_exit[exit] += 1
                            break
                """

                total += len(y)

            print(f'{year}-{month}: Accuracy: {100*correct[0]/total:.2f} | {100*correct[1]/total:.2f} | {100*correct[2]/total:.2f}')

            accuracies = ' | '.join([ f'{100*acc/total:.6}% ' for acc in correct ])
            
            print(f'Accuracies per exit (for all dataset): {accuracies}')

        seq += 1
