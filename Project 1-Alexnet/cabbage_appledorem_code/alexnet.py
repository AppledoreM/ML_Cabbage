#!/usr/bin/env python
# coding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import numpy



class Alexnet(nn.Module):
    def __init__(self, num_classes, fc_size = 4096):
        super(Alexnet, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # Block 2
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # Block 3
            nn.Conv2d(256, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            # Block 4
            nn.Conv2d(384, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            # Block 5
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        self.classifer = nn.Sequential(
            # Block 6
            nn.Linear(256 * 5 * 5, fc_size),
            # Block 7
            nn.Linear(fc_size, fc_size),
            # Block 8
            nn.Linear(fc_size, num_classes),
        )
    

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 5 * 5)
        return self.classifer(x)


