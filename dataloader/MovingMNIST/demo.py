import os
import sys
import numpy as np
import matplotlib.pyplot as plt

parent_path = os.path.dirname(os.path.dirname(os.getcwd()))
print(os.getcwd())
if 'MovingMNIST' in os.getcwd():
    os.chdir(parent_path)
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from dataloader.MovingMNIST.MovingMNIST import MovingMNIST

root = './data'
if not os.path.exists(root):
    os.mkdir(root)


train_set = MovingMNIST(root=os.path.join('.', 'data','mnist'), train=True, download=True,
                        transform=transforms.Compose([transforms.ToTensor(),]),
                        target_transform=transforms.Compose([transforms.ToTensor(),]))
test_set = MovingMNIST(root=os.path.join('.', 'data','mnist'), train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor(),]),
                        target_transform=transforms.Compose([transforms.ToTensor(),]))

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))


for seq, seq_target in train_loader:
    print('--- Sample')
    print('Input:  ', seq.shape)
    print('Target: ', seq_target.shape)
    break