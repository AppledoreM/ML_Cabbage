#!/usr/bin/env python
# coding=utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from alexnet import Alexnet
import matplotlib.pyplot as plt
import argparse

def train_argparse(train_arguments):

    train(train_arguments.model_save_dir,
          train_arguments.dataset_dir,
          train_arguments.batch_size,
          train_arguments.learning_rate,
          train_arguments.num_worker,
          train_arguments.epochs,
          train_arguments.load_model_dir
         )




def train(model_save_dir, dataset_dir, batch_size, lr, num_worker, num_epoch, load_model_dir):
    
    # Initialize a list of transformation
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Set CIFAR10 Dataset
    train_dataset = datasets.CIFAR10(dataset_dir, train = True, download = True, transform = transform)
    # Set CIFAR10 DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_worker)
    
    # Set Alexnet Model
    model = Alexnet(10)
    if load_model_dir is not None:
        model.load_state_dict(torch.load(load_model_dir))
    model.cuda()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    model.train()
    
    loss_lis = []

    for epoch in range(num_epoch):
        tot_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = Variable(x.cuda())
            y = Variable(y.cuda())
            optimizer.zero_grad()
            output = model(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step() 
            tot_loss = tot_loss + loss.data.item()

            if batch_idx % 100 == 0:
                print('Train Epoch: {}  {:.2f}% Percent Finished. Current Loss: {:.6f}'.format(
                   epoch + 1,
                    100 * batch_idx / len(train_loader),
                    tot_loss
                ))
        print('Epoch {} Finished! Total Loss: {:.2f}'.format(epoch + 1, tot_loss))
    
        loss_lis.append(tot_loss)

    torch.save(model.state_dict(), model_save_dir)
    x = [x + 1 for x in range(num_epoch)]
    plt.plot(x, loss_lis)
    plt.xlabel("Epoch Number")
    plt.ylabel("Total Loss Per Epoch")
    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--batch_size", type = int, default = 64, help = "Set the training batch size.")
    parser.add_argument("--num_worker", type = int, default = 2, help = "Set the number of thread used in loading data.")
    parser.add_argument("--learning_rate", type = float, default = 0.001, help = "Set the learning rate.")
    parser.add_argument("--dataset_dir", type = str, default = "../../datasets/", help = "Set the location of the dataset.")
    parser.add_argument("--model_save_dir", type = str, default = "./model/alexnet.pth", help = "Set the location of the model saved") 
    parser.add_argument("--epochs", type = int, default = 100, help = "Set the number of training epoch.")
    parser.add_argument("--load_model_dir", type = str, default = None, help = "Set if you want to load a model to train.")
    args = parser.parse_args()

    train_argparse(args)


