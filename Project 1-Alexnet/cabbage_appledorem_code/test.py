#!/usr/bin/env python
# coding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from alexnet import Alexnet


def imshow(img):
    img = img / 2 + 0.5
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def test(batch_size):
    transform = transforms.Compose(
        [
         transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    test_dataset = torchvision.datasets.CIFAR10("../data/", download = True, train = False, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    model = Alexnet(10).cuda()
    model.load_state_dict(torch.load("./model/alexnet.pth"))

    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    with torch.no_grad():
        tot = 0
        correct = 0
        for x, y in test_loader:
            x = x.cuda()
            y = y.cuda()

            out = model(x)
            _, predicted = torch.max(out.data, 1)
            
            print(y)
            
            for j in range(batch_size):
                print("{}th Picture. Predicted: {}. Actual: {}".format(j, classes[predicted[j]], classes[y.cpu()[j]]))
                tot = tot + 1
                if predicted[j] == y.cpu()[j]:
                    correct = correct + 1
            print('--------------------------------------------')

        print("Total Picture:{}. Correct Predict: {}. Percentage: {:.2f}%".format(tot, correct, correct * 100 / tot))

if __name__ == "__main__":
    test(10)
    



