#!/usr/bin/env python
# coding=utf-8

import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from vgg import VGG


# Yes, it is called Trainer
class Trainer:
    def __init__(self):
        self.parser = argparse.ArgumentParser();
        self.parser.add_argument("--batch_size", type = int, default = 64, help = "Set the training batch size. Default: 64.")
        self.parser.add_argument("--num_worker", type = int, default = 2, help = "Set the number of thread used in loading data. Default: 2.")
        self.parser.add_argument("--learning_rate", type = float, default = 0.001, help = "Set the learning rate. Default: 0.001.")
        self.parser.add_argument("--dataset_dir", type = str, default = "../../datasets/", help = "Set the location of the dataset. Default: '../data/'")
        self.parser.add_argument("--model_save_dir", type = str, default = "./model/", help = "Set the location of the model saved. Default: './model/'") 
        self.parser.add_argument("--epochs", type = int, default = 100, help = "Set the number of training epoch. Default: 100") 
        self.parser.add_argument("--load_model_dir", type = str, default = None, help = "Set if you want to load a model to train. Default: None") 
        self.parser.add_argument("--model_type", type = str, default = None, help = "Set model type if there are multiple plan for a model. Default: None")

        # Parse arguments passed in
        self.parse_arguments(self.parser.parse_args())

    def get_parser(self):
        return self.parser

    def parse_arguments(self, args):
        self.batch_size = args.batch_size
        self.num_worker = args.num_worker
        self.learning_rate = args.learning_rate
        self.dataset_dir = args.dataset_dir
        self.model_save_dir = args.model_save_dir
        self.epochs = args.epochs
        self.load_model_dir = args.load_model_dir
        self.model_type = args.model_type

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGG(self.model_type, True).to(device)


        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = self.learning_rate)


        start_epoch = 0
        end_epoch = self.epochs
        average_loss_list = []
        writer = SummaryWriter('logs')

        if self.load_model_dir is not None:
            checkpoint = torch.load(self.load_model_dir)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = checkpoint['optimizer_state_dict']
            start_epoch += checkpoint['epoch']
            end_epoch += checkpoint['epoch']
            average_loss_list = checkpoint['average_loss_list']
            for idx, loss in enumerate(average_loss_list):
                writer.add_scalar("Training Loss Average", loss, idx + 1)



        transform_ops = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.CIFAR100(self.dataset_dir, train = True, download = True, transform = transform_ops)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_worker)


        model.train()


        running_loss = 0 
        running_idx = 0
        running_temp_idx = 0
        
        for epoch in range(start_epoch, end_epoch):
            total_loss = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                x = Variable(x.cuda())
                y = Variable(y.cuda())
                optimizer.zero_grad()
                
                predicted = model(x)
                loss = loss_function(predicted, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.data.item()

                running_loss += loss.data.item()
                running_temp_idx += 1
                if batch_idx % 100 == 0:
                    print('Train Epoch: {}  {:.2f}% Percent Finished. Current Loss: {:.6f}'.format(
                        epoch + 1,
                        100 * batch_idx / len(train_loader),
                        total_loss
                    ))

                if running_temp_idx % 50 == 0:
                    running_idx += 1
                    writer.add_scalar("Running Loss", running_loss / 100, running_idx)
                    runninng_temp_idx = 0
                    running_loss = 0

            writer.add_scalar("Training Loss Average", total_loss / len(train_loader), epoch + 1)
            print('Epoch {} Finished! Total Loss: {:.2f}'.format(epoch + 1, total_loss))
            average_loss_list.append(total_loss / len(train_loader))
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch' : epoch + 1,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'average_loss_list' : average_loss_list

                }, self.model_save_dir + "vgg-checkpoint-{}.pth".format(epoch + 1))
                
        torch.save({
            'epoch' : epoch + 1,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'average_loss_list' : average_loss_list
        }, self.model_save_dir + "vgg.pth".format(epoch + 1))



if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()










