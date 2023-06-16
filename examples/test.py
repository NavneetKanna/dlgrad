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
from dlgrad.afu import ReLU
from nn.training import train, test, plot_metrics 
from dlgrad.graph import save_graph
from dlgrad import optim
import time
import numpy as np


# kernprof -l -v --unit 1e-3 test.py
# https://medium.com/geekculture/a-look-under-the-hood-of-pytorchs-recurrent-neural-network-module-47c34e61a02d
# python -m unittest discover -v


# TODO: Get rid of helper.py






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
        x = ReLU(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = ReLU(x)
        x = self.pool2(x)
        x = Tensor.flatten(x) # flatten all dimensions except batch
        x = self.fc1(x)
        x = ReLU(x)
        x = self.fc2(x)
        x = ReLU(x)
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
    

    local_dict = None
    start_time = time.perf_counter()
    for epoch in range(epochs):
            print(f"epoch {epoch+1}")
            fashion_mnist_dataset.reset_idx()
            local_dict = train(net, fashion_mnist_dataset, x_train, y_train, BS, optimizer)
    fashion_mnist_dataset.delete_data()

    import matplotlib.pyplot as plt
    l = []
    # for j, i in enumerate(local_dict):
    #     print(f"mean {i.tensor.mean()} std {i.tensor.std()} saturated {(np.absolute(i.tensor) > 0).mean()*100}%")
    #     hy, hx = np.histogram(i.tensor, density=True)
    #     plt.plot(hx[:-1], hy)
    #     l.append(f"layer {j+1} relu")
    # plt.legend(l)
    # plt.show()

    # print("Grad")
    # for j, i in enumerate(local_dict):
    #     print(f"mean {i.grad.mean()} std {i.grad.std()}")
    #     hy, hx = np.histogram(i.grad, density=True)
    #     plt.plot(hx[:-1], hy)
    #     l.append(f"layer {j+1} relu")
    # plt.legend(l)
    # plt.show()
        
              
         

    end_time = time.perf_counter()
    dot_time = end_time - start_time
    print(f"time = {dot_time}")

if __name__ == '__main__':
    main()

