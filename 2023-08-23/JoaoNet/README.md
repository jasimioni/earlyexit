# AlexNet

## models/AlexNet.py

Some AlexNet adapted - MNIST, CIFAR10

- AlexNetCIFAR10Layers

Layers used to build the CIFAR10 DNNs

- AlexNetCIFAR10ee1

Simulate a regular DNN with only ee1

- AlexNetCIFAR10ee2

Simulate a regular DNN with only ee2

- AlexNetCIFAR10

The regular AlexNet for CIFAR10 using the layers (same as exit3)

- AlexNetWithExistsCIFAR10

Magic Model with early exits and specific training code

## utils/functions.py

- `train_model` to train all exists together
- `train_exit` to train specific exit
- `show_exit_status` to display statistics

## alexnet.py

Code to train / evaluate model. Need to comment / uncomment things to decide what to do

## conda/earlyexit.yaml

Dependencies for conda, if using it:

- requests
- matplotlib
- pytorch
- torchvision
