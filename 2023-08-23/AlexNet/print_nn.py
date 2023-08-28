#training, testing for branchynet-pytorch version
#testing fit with onnx

from models.Branchynet import Branchynet, ConvPoolAc

from sys import argv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets
import matplotlib.pyplot as plt

import os
import numpy as np
from datetime import datetime as dt

model = Branchynet(exit_threshold=0.999)

print(model.backbone)
print(model.exits)
