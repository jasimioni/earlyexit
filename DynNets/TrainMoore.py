#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchinfo import summary
import numpy as np
import pandas as pd

from datetime import datetime as dt
from datetime import datetime
import os
import time
import sys
sys.path.append('..')

from utils.functions import *
import models.MooreNet

net = 'TryNet13'

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

set_writer(f'runs/train_{net}_{dt_string}')
writer = get_writer()

def log(file, msg):
    print(msg)
    # print(msg, file=file)

def train(model, device, train_loader, file_prefix, epochs=20):
    lr=0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # log_file = open(f'{file_prefix}.log', 'w')
    log_file = None
    # save_file = f'{file_prefix}.save'

    model_summary = summary(model)
    log(log_file, f'Training {model} in {device}')
    log(log_file, str(model_summary))

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

            # cnf = torch.mean(torch.max(nn.functional.softmax(y_pred, dim=-1), 1)[0]).item()

            writer.add_scalar(f'Accuracy', trn_cor.item()*100/trn_cnt, seq)

            if (b-1)%10 == 0:
                log(log_file, f'Epoch: {i:2} Batch: {b:3} Loss: {loss.item():4.4f} Accuracy Train: {trn_cor.item()*100/trn_cnt:2.3f}%')

            seq += 1

    log(log_file, f'\nDuration: {time.time() - start_time:.0f} seconds')
    log_file.close()
   
    #save_dict = { 
    #    'model_state_dict': model.state_dict()
    #}    
    #torch.save(save_dict, save_file) 

batch_size = 5000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_data   = CustomMawiDataset(year='2016', month='01', as_matrix=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# nets = [ 'TryNet14', 'TryNet11', 'TryNet12', 'TryNet13' ]
nets = [ net ]

for net in nets:
    print(net)
    m_class = getattr(models.TryNets, net)
    model = m_class().to(device)
    train(model, device, train_loader, net)

# test_data    = CustomMawiDataset(year='2016', month='03', as_matrix=True)

# summary(model, (1, 1, 7, 7))

# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

