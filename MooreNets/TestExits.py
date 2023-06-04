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
model.eval()
model.set_measurement_mode()
# Run one inference to "load" everything
# with torch.no_grad():
#    model(torch.rand(1, 1, 8, 8).to(device))

thresholds = [ 0.9, 0.9, 0 ]

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
            times = { 'bb' : [0, 0, 0], 'ex' : [0, 0, 0] }
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
                    times['bb'][exit] += result[1] * len(result[0])
                    times['ex'][exit] += result[2] * len(result[0])
                
                    cnf, predicted = torch.max(nn.functional.softmax(result[0], dim=-1), 1)

                    c_results.append([cnf, predicted])
                            
                    correct[exit] += (predicted == y).sum()

                for i in range(len(y)):
                    for exit, res in enumerate(c_results):
                        if res[0][i] > thresholds[exit]:
                            chosen_exit[exit] += 1
                            if res[1][i] == y[i]:
                                correct_exit[exit] += 1
                            break

                total += len(y)

            print(f'{year}-{month}: Accuracy: {100*correct[0]/total:.2f} | {100*correct[1]/total:.2f} | {100*correct[2]/total:.2f}')

            mean_time = []
            mean_time.append((times['bb'][0] + times['ex'][0])/total)
            mean_time.append((times['bb'][0] + times['bb'][1] + times['ex'][1])/total)
            mean_time.append((times['bb'][0] + times['bb'][1] + times['bb'][2] + times['ex'][2])/total)
            mean_times = ' | '.join([ f'{1000*mt:.4} ms' for mt in mean_time ])
            accuracies = ' | '.join([ f'{100*acc/total:.6}% ' for acc in correct ])
            chosen_exits  = ' | '.join([ f'{100*x/total:.2f}%' for x in chosen_exit ])
            exit_accuracy = []
            for i, n_chosen in enumerate(chosen_exit):
                if n_chosen > 0:
                    exit_accuracy.append(f'{100*correct_exit[i]/n_chosen:.2f} %')
                else:
                    exit_accuracy.append('NaN')
            exit_accuracy = ' | '.join(exit_accuracy)
            thresholds_str = ' | '.join([ f'{t}' for t in thresholds ])
            
            t_mean_time = 0
            for c, mt in zip(chosen_exit, mean_time):
                t_mean_time += c * mt
            t_mean_time /= total

            print(f'Thresholds: {thresholds_str}')
            print(f'Mean times per exit: {mean_times}')
            print(f'Accuracies per exit (for all dataset): {accuracies}')
            print(f'Rate of exit chosen: {chosen_exits}')
            print(f'Accuracy per exit (when chosen): {exit_accuracy}')
            print(f'Overall Accuracy: {100 * sum(correct_exit)/total:.2f}%')        
            print(f'Mean time: {1000 * t_mean_time:.4f} ms')

        seq += 1
