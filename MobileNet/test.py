#training, testing for branchynet-pytorch version
#testing fit with onnx

from models.Branchynet import Branchynet, ConvPoolAc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib

import os
import numpy as np
from datetime import datetime as dt


model = Branchynet()

print(model.parameters())
print(model.backbone.parameters())
print(model.exits[-1].parameters())


