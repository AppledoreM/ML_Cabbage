#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torchvision
import numpy
import argparse


# This is modified to fit the training time and CIFAR100 Model
cfg = {
        'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
        'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
        'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
        'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, model_type = 'E', _batch_norm = False, _class_count = 100):
        super(VGG, self).__init__()
        
        self.features = self.make_layers(model_type, _batch_norm)
        self.classifer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), 
            nn.ReLU(inplace = True), 
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, _class_count)
        )

    def make_layers(self, model_type, _batch_norm):
        config = cfg[model_type]
        input_channel = 3
        layer_list = []
        for layer in config:
            if layer == 'M':
                 layer_list += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
                 continue

            layer_list += [nn.Conv2d(input_channel, layer, kernel_size = 3, padding = 1)]
            if _batch_norm:
                layer_list += [nn.BatchNorm2d(layer)]
            layer_list += [nn.ReLU(inplace = True)]
            input_channel = layer
        return nn.Sequential(*layer_list)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifer(x) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model_type", type = str, default = 'E', help = "Set the type of model structure to be printed. Default Setting: VGG-19 Plan 'E'")
    parser.add_argument("--batch_norm", type = bool, default = False, help = "Set if batch normalization layers are enabled. Default Setting: False")
    parser.add_argument("--num_class", type = int, default = 100, help = "Set the number of classes the model is going to classify. Default Setting: 100")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG(args.model_type, args.batch_norm, args.num_class).to(device)
    print(model)




