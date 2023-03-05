from models.AlexNet import AlexNetCIFAR10, AlexNetCIFAR10ee1, AlexNetCIFAR10ee2
from models.Branchynet import Branchynet

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

model = AlexNetCIFAR10ee1()
print(model)
