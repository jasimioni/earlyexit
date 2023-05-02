#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchinfo import summary
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime as dt
import os
import time
import sys
sys.path.append('..')

from utils.functions import *
import models.TryNets

set_writer('test_nets/tst01')
writer = get_writer()

batch_size = 50000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

nets = [ 'TryNet14', 'TryNet11', 'TryNet12', 'TryNet13' ]
c_models = {}
for net in nets:
    print(net)
    m_class = getattr(models.TryNets, net)
    model = m_class().to(device)

    save_file = f'{net}.save'

    checkpoint = torch.load(save_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    c_models[net] = model

seq = 0
for year in (2016, 2017, 2018, 2019):
    year = str(year)

    for month in range(12):
        month = f'{month+1:02d}'

        data = CustomMawiDataset(year=year, month=month, as_matrix=True)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        for net in nets:
            model = c_models[net]
            with torch.no_grad():
                tst_cor = 0
                tst_cnt = 0
                for b, (X_test, y_test) in enumerate(loader):
                    X_test = X_test.to(device)
                    y_test = y_test.to(device)
                    y_pred = model(X_test)            
                    predicted = torch.max(y_pred.data, 1)[1]
                    batch_cor = (predicted == y_test).sum()
                    tst_cor += batch_cor
                    tst_cnt += len(predicted)
                print(f'{year}-{month}: {net} - Accuracy Test: {tst_cor.item()*100/tst_cnt:2.3f}%')
                writer.add_scalar(f'Accuracy {net}', tst_cor.item()*100/tst_cnt, seq)

        seq += 1