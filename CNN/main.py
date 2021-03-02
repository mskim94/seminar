# -*- coding: utf-8 -*-
# +
'''Train Standard Dogs Data with PyTorch'''
import os
import argparse

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader

from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import models
from utils import progress_bar
from utils import GradualWarmupScheduler
# -

model_names = sorted(name for name in models.__dict__ 
                     if name.islower() and not name.startswith("__") 
                     and callable(models.__dict__[name]))


optim_names = sorted(name for name in optim.__dict__ 
                     if not name.startswith("__") and 
                     callable(optim.__dict__[name]))


def get_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Stanford Dogs Dataset Training")

    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--optimizer', default='Adam', choices=optim_names, help='optimizer')
    parser.add_argument('--model', default='resnet18', choices=model_names, help='model')
    parser.add_argument('--pretrained', action='store_true', help='pretrained')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--num_classes', default=1000, type=int, help='number of classes')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', help='nesterov')
    
    args = parser.parse_args()
    
    return args


args = get_arguments()

# +
# import easydict

# args = easydict.EasyDict({
#         "epochs": 200,
#         "batch_size": 64,
#         "lr": 0.001,
#         "optimizer": 'Adam',
#         "model": 'resnet18',
#         "pretrained": True,
#         "resume": True,
# })
# -

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0


# Data
print('==> Preparing Data...')


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255), 
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

DATASET_PATH = "/home/work/cnn_oop/data"

train_data = datasets.ImageFolder(DATASET_PATH + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(DATASET_PATH + '/test', transform=test_transforms)

batch_size = args.batch_size
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=2) # num_workers는 많이 잡아
test_iter = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, num_workers=2) # test는 Shuffle x

print("Preparing dataset done!")


# +
# Model 
print("==> Building Model...")

if args.model == "efficientnet":
    net = models.EfficientNet.from_pretrained("efficientnet-b4", num_classes=args.num_classes)
else:
    net = models.__dict__[args.model](pretrained=args.pretrained, num_classes=args.num_classes)

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
# -

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}_{}_{}.pth'.format(args.model, args.optimizer, args.pretrained))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# +
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=args.lr)
# optimizer = optim.__dict__[args.optimizer](net.parameters(), lr=args.lr)
optimizer = optim.__dict__[args.optimizer]

if optimizer == torch.optim.SGD:
    optimizer = optimizer(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                         nesterov=args.nesterov)
else:
    optimizer = optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)


# -

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_iter):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_iter), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_iter):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print("Test acc: %.3f" %(acc))
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_{}_{}.pth'.format(args.model, args.optimizer, args.pretrained))
        best_acc = acc


# + endofcell="--"
for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    scheduler_warmup.step()

# for epoch in range(start_epoch, start_epoch+200):
#      train(epoch)
#      test(epoch)
#      scheduler.step()

"""
if __name__ == "__main__":
    args = get_arguments()
    
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
# -

"""
# --
