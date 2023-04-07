#!/usr/bin/env python3

import argparse
import sys
sys.path.append('..')

from itertools import chain

from models.JoaoNet import *
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

import os
import numpy as np
from datetime import datetime as dt

parser = argparse.ArgumentParser(description='Print results for model')

parser.add_argument("--load", help="Filename to load from")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.ToTensor()

# model = JoaoNetCIFAR10().to(device)
# checkpoint = torch.load('saves/JoaoNetCIFAR10_CrossEntropy_2023-04-06')
model = JoaoNetWithExitsCIFAR10().to(device)
# checkpoint = torch.load('saves/JoaoNetWithExitsCIFAR10_all_CrossEntropy_2023-04-06')
checkpoint = torch.load('saves/JoaoNetWithExitsCIFAR10_all_ConfidenceOnCorrect_2023-04-06')
model.load_state_dict(checkpoint['model_state_dict'])

batch_size = 600

train_data   = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_data    = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

thresholds = [ 0.5, 0.5, 0 ]

model.eval()
with torch.no_grad():
    # Run on inference to "load" everything
    model(torch.rand(1, 3, 32, 32).to(device))
    if hasattr(model, 'exits'):
        model.set_measurement_mode()
        total = 0
        correct = 0
        times = { 'bb' : [0, 0, 0], 'ex' : [0, 0, 0] }
        correct = [0, 0, 0]
        correct_exit = [0, 0, 0]
        chosen_exit  = [0, 0, 0]
        for i, (X, y) in enumerate(chain(train_loader, test_loader)):
            X = X.to(device)
            y = y.to(device)
   
            results = model(X)
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
            
        mean_time = []
        mean_time.append((times['bb'][0] + times['ex'][0])/total)
        mean_time.append((times['bb'][0] + times['bb'][1] + times['ex'][1])/total)
        mean_time.append((times['bb'][0] + times['bb'][1] + times['bb'][2] + times['ex'][2])/total)
        mean_times = ' | '.join([ f'{1000*mt:.4} ms' for mt in mean_time ])
        accuracies = ' | '.join([ f'{100*acc/total:.6}% ' for acc in correct ])
        chosen_exits  = ' | '.join([ f'{100*x/total:.2f}%' for x in chosen_exit ])
        exit_accuracy = ' | '.join([ f'{100*correct_exit[i]/chosen_exit[i]:.2f}%' for i in range(len(chosen_exit))])
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
    else:
        total = 0
        correct = 0
        time_total = 0
   
        for i, (X, y) in enumerate(chain(train_loader, test_loader)):
            X = X.to(device)
            y = y.to(device)

            st = time.process_time()   
            y_pred = model(X)
            ed = time.process_time()    
            
            time_total += ( ed - st ) * len(y)
            
            predicted = torch.max(y_pred, 1)[1]
            
            correct += (predicted == y).sum()
            total   += len(y_pred)
        
        print(f'Total: {total}, Correct: {correct}, Accuracy: {100 * correct/total:.2f}, Average Inference Time: {1000*time_total/total:.6f} ms')