#!/usr/bin/env python3

from models.JoaoNet import JoaoNetCIFAR10Layers

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

model = JoaoNetCIFAR10Layers()

print(model.backbone)
print(model.exits)
