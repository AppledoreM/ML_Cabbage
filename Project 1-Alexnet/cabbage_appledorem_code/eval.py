#!/usr/bin/env python
# coding=utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import numpy
import PIL
import argparse
from alexnet import Alexnet


def readimg(img_loc):
    return PIL.Image.open(img_loc)

def eval(image_loc):
    x = readimg(image_loc)
    transform = transforms.Compose(
        [
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    x = transform(x)
    x = [x]
    x = torch.stack(x).cuda()

    model =  Alexnet(10).cuda()
    model.load_state_dict(torch.load("./model/alexnet.pth"))

    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    with torch.no_grad():
        out = model(x)
        _, predicted = torch.max(out.data, 1)

        print(predicted)
        print("Predicted class: {}".format(classes[predicted]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--image", type = str, required = True, help = "Set the image location for evaluation.")
    args = parser.parse_args()

    eval(args.image)








