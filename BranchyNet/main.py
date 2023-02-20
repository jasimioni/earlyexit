#training, testing for branchynet-pytorch version
#testing fit with onnx

from models.Branchynet import Branchynet, ConvPoolAc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from datetime import datetime as dt

def train_backbone(model, train_dl, valid_dl, save_path, epochs=50,
                    loss_f=nn.CrossEntropyLoss(), opt=None):
    if opt is None:
        lr = 0.001
        exp_decay_rates = [0.99, 0.999]
        backbone_params = [
                {'params': model.backbone.parameters()},
                {'params': model.exits[-1].parameters()}
                ]

        opt = optim.Adam(backbone_params, betas=exp_decay_rates, lr=lr)

    best_val_loss = [1.0, '']
    for epoch in range(epochs):
        model.train()
        print("Starting epoch:", epoch+1, end="... ", flush=True)

        for xb, yb in train_dl:
            results = model(xb)
            loss = loss_f(results[-1], yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            valid_losses = np.sum(np.array(
                    [[loss_f(exit, yb) for exit in model(xb)]
                        for xb, yb in valid_dl]), axis=0)

        val_loss_avg = valid_losses[-1] / len(valid_dl)
        print("V Loss:", val_loss_avg)
        savepoint = save_model(model, save_path, file_prefix='backbone-'+str(epoch+1), opt=opt)

        if val_loss_avg < best_val_loss[0]:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint
    print("BEST VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
    return savepoint

def train_exits(model, epochs=100):
    return

def train_joint(model, train_dl, valid_dl, save_path, opt=None,
                loss_f=nn.CrossEntropyLoss(), backbone_epochs=50,
                joint_epochs=100, pretrain_backbone=True):

    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")

    if pretrain_backbone:
        print("PRETRAINING BACKBONE FROM SCRATCH")
        folder_path = 'pre_Trn_bb_' + timestamp
        best_bb_path = train_backbone(model, train_dl,
                valid_dl, os.path.join(save_path, folder_path),
                epochs=backbone_epochs, loss_f=loss_f)
        print("LOADING BEST BACKBONE")
        load_model(model, best_bb_path)
        print("JOINT TRAINING WITH PRETRAINED BACKBONE")

        prefix = 'pretrn-joint'
    else:
        print("JOINT TRAINING FROM SCRATCH")
        folder_path = 'jnt_fr_scrcth' + timestamp
        prefix = 'joint'

    spth = os.path.join(save_path, folder_path)

    if opt is None:
        lr = 0.001
        exp_decay_rates = [0.99, 0.999]
        opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

    best_val_loss = [[1.0,1.0], ''] #TODO make sure list size matches num of exits

    for epoch in range(joint_epochs):
        model.train()
        print("starting epoch:", epoch+1, end="... ", flush=True)

        for xb, yb in train_dl:
            results = model(xb)

            losses = [weighting * loss_f(res, yb)
                        for weighting, res in zip(model.exit_loss_weights,results)]

            opt.zero_grad()
            for loss in losses[:-1]:
                loss.backward(retain_graph=True)
            losses[-1].backward()

            opt.step()

        model.eval()
        with torch.no_grad():
            valid_losses = np.sum(np.array(
                    [[loss_f(exit, yb) for exit in model(xb)]
                        for xb, yb in valid_dl]), axis=0)


        val_loss_avg = valid_losses / len(valid_dl)
        print("v loss:", val_loss_avg)
        savepoint = save_model(model, spth, file_prefix=prefix+'-'+str(epoch+1), opt=opt)

        el_total=0.0
        bl_total=0.0
        for exit_loss, best_loss in zip(val_loss_avg,best_val_loss[0]):
            el_total+=exit_loss
            bl_total+=best_loss
        if el_total < bl_total:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint

    print("BEST* VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
    return

def pull_mnist_data(batch_size=64):
    tfs = transforms.Compose([
        transforms.ToTensor()
        ])

    mnist_train_dl = DataLoader( torchvision.datasets.MNIST('../data/mnist',
                                    download=True, train=True, transform=tfs),
                batch_size=batch_size, drop_last=True, shuffle=True)

    mnist_valid_dl = DataLoader( torchvision.datasets.MNIST('../data/mnist',
                                    download=True, train=False, transform=tfs),
                batch_size=batch_size, drop_last=True, shuffle=True)

    return mnist_train_dl, mnist_valid_dl

def save_model(model, path, file_prefix='', seed=None, epoch=None, opt=None, loss=None):
    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    filenm = file_prefix + '-' + timestamp
    save_dict ={'timestamp': timestamp,
                'model_state_dict': model.state_dict()
                }

    if seed is not None:
        save_dict['seed'] = seed
    if epoch is not None:
        save_dict['epoch'] = epoch
        filenm += f'{epoch:03d}'
    if opt is not None:
        save_dict['opt_state_dict'] = opt.state_dict()
    if loss is not None:
        save_dict['loss'] = loss

    if not os.path.exists(path):
        os.makedirs(path)

    filenm += '.pth'
    file_path = os.path.join(path, filenm)

    torch.save(save_dict, file_path)

    print("Saved to:", file_path)
    return file_path


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

def main():
    model = Branchynet()
    print("Model done")

    batch_size = 512
    train_dl, valid_dl = pull_mnist_data(batch_size)
    print("Got training and test data")

    loss_f = nn.CrossEntropyLoss()
    print("Loss function set")

    bb_epochs = 5
    jt_epochs = 10
    path_str = 'outputs/'

    train_joint(model, train_dl, valid_dl, path_str, backbone_epochs=bb_epochs,
            joint_epochs=jt_epochs, loss_f=loss_f, pretrain_backbone=True)

if __name__ == "__main__":
    main()
