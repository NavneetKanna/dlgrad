import sys
import os
os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad/")
# os.chdir(r"C:/Users/navne/Documents/vs_code/dlgrad")
sys.path.append(os.getcwd())
from dlgrad.mlp import MLP
from dlgrad.conv import Conv2d, MaxPool2d
# from datasets.fetch_mnist import MNIST
from datasets.fetch_cifar10 import CIFAR10
from datasets.fetch_fashion_mnist import MNIST
# from datasets.fetch_mnist_cnn import MNIST
from dlgrad.tensor import Tensor
from dlgrad.mlp import MLP
# from dlgrad.afu import ReLU
from nn.training import train, test, plot_metrics 
from dlgrad.graph import save_graph
from dlgrad import optim
import time
import numpy as np


# kernprof -l -v --unit 1e-3 test.py
# python -m unittest discover -v


# TODO: Get rid of helper.py
# TODO: Should afu and loss come in tensor.py ?


class Net:
    def __init__(self):
        self.conv1 = Conv2d(3, 6, 5)
        self.pool1 = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        self.pool2 = MaxPool2d(2, 2)
        self.fc1 = MLP(16*5*5, 120, bias=True)
        self.fc2 = MLP(120, 84, bias=True)
        self.fc3 = MLP(84, 10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.ReLU()
        x = self.pool1(x)
        x = self.conv2(x)
        x = x.ReLU()
        x = self.pool2(x)
        x = Tensor.flatten(x) # flatten all dimensions except batch
        x = self.fc1(x)
        x = x.ReLU()
        x = self.fc2(x)
        x = x.ReLU()
        x = self.fc3(x)
        return x

def main():
    epochs = 2 
    BS = 32 
    lr = 1e-3 
    
    net = Net()

    start_time = time.perf_counter()
    optimizer = optim.SGD(net, lr)

    fashion_mnist_dataset = CIFAR10()
    x_train, y_train = fashion_mnist_dataset.get_train_data()
    

    start_time = time.perf_counter()
    for epoch in range(epochs):
            print(f"epoch {epoch+1}")
            fashion_mnist_dataset.reset_idx()
            train(net, fashion_mnist_dataset, x_train, y_train, BS, optimizer)
    
    fashion_mnist_dataset.delete_data()
    end_time = time.perf_counter()
    dot_time = end_time - start_time
    print(f"time = {dot_time}")

if __name__ == '__main__':
    main()

