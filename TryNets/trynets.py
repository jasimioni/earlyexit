#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchinfo import summary
import numpy as np
import pandas as pd

from datetime import datetime as dt
import os
import time
import sys
sys.path.append('..')

from utils.functions import *
from models.TryNets import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_data   = CustomMawiDataset(year='2019', month='01', as_matrix=False)
test_data    = CustomMawiDataset(year='2019', month='02', as_matrix=False)

model = TryNet08().to(device)
summary(model, (1, 48))
summary(model)

epochs = 5
batch_size = 300

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

lr=0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

import time
start_time = time.time()

seq = 0
for i in range(epochs):
    trn_cor = 0
    trn_cnt = 0
    tst_cor = 0
    tst_cnt = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        b+=1

        y_pred = model(X_train)

        # print(y_pred)
        # int(y_train)

        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.max(y_pred.data, 1)[1]
        batch_cor = (predicted == y_train).sum()
        trn_cor += batch_cor
        trn_cnt += len(predicted)

        cnf = torch.mean(torch.max(nn.functional.softmax(y_pred, dim=-1), 1)[0]).item()

        if (b-1)%10 == 0:
            print(f'Epoch: {i:2} Batch: {b:3} Loss: {loss.item():4.4f} Accuracy Train: {trn_cor.item()*100/trn_cnt:2.3f}%')

        seq += 1

print(f'\nDuration: {time.time() - start_time:.0f} seconds')
