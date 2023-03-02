#training, testing for branchynet-pytorch version
#testing fit with onnx

from models.AlexNet import AlexNetMNIST

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

model = AlexNetMNIST()
