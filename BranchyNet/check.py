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

try:
    path = argv[1]
    if not os.path.isfile(path):
        raise Exception(f'{path} is not a valid file')
except Exception as e:
    if type(e).__name__ == 'IndexError':
        print(f"Usage: {argv[0]} <filename>")
    else:
        print(e)
    exit()

model = Branchynet(exit_threshold=0.7)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
model.set_fast_inf_mode()

mnist_dl = DataLoader(torchvision.datasets.MNIST('../data/mnist',
                      download=True, train=False, transform=ToTensor()),
                      batch_size=1, drop_last=True, shuffle=False)

# mnistiter = iter(mnist_dl)

count = 0
for xb, yb in mnist_dl:
    # print(xb)
    output, exit = model(xb)
    output = output.detach().numpy()[0]
    predicted = max(range(len(output)), key=output.__getitem__)
    print(f'Predicted: {predicted}, Correct: {yb.item()}, Exit: {exit}')
    count += 1
    #if count > 100:
    #    break


'''
test_data = datasets.MNIST(
    '../data/mnist',
    train=False,
    download=True,
    transform=ToTensor()
)

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(test_data), size=(1,)).item()
    img, label = test_data[sample_idx]
    result = model(img)
    print(result)
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''