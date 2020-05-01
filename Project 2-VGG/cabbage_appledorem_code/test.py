#!/usr/bin/env python
# coding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from vgg import VGG

def test(test_model_dir, batch_size, mean = (0.5,0.5,0.5), std = (0.5, 0.5, 0.5), model = None):
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)

        ]

    )
    test_dataset = torchvision.datasets.CIFAR100("../../datasets/", download = True, train = False, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    if model is None:
        state_dict = torch.load(test_model_dir)
        model = VGG(state_dict['model_type'], True).cuda()
        model.load_state_dict(state_dict["model_state_dict"])

    classes = []
    accuracy = 0.0
    with torch.no_grad():
        tot = 0
        correct = 0

        for idx, (x, y) in enumerate(test_loader):
            x = x.cuda()
            y = y.cuda()

            out = model(x)
            _, predicted = torch.max(out.data, 1)

            for j in range(len(predicted)):
                tot = tot + 1
                if predicted[j] == y.cpu()[j]:
                    correct = correct + 1

            print("Test Percent Finished: {}%.".format(idx * 100/ len(test_loader)))

        print("{} th Picture Predicted: Correct Predict: {}. Correct Percentage: {:.2f}%".format(tot, correct, correct * 100 / tot))
        accuracy = correct / tot * 100
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_model_dir", type = str, required = True, help = "Select which model to test.")
    args = parser.parse_args()
    test(args.test_model_dir, 100)


