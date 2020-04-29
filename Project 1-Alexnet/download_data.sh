#!/bin/bash
mkdir ../datasets
wget -O data/cifar-100.tar.gz "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
cd ../datasets/
tar -xvzf cifar-100.tar.gz
cd ../Project 1-Alexnet

