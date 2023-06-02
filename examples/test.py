import sys
import os
os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad/")
# os.chdir(r"C:/Users/navne/Documents/vs_code/dlgrad")
sys.path.append(os.getcwd())
from dlgrad.mlp import MLP
from dlgrad.conv import Conv2d, MaxPool2d
# from datasets.fetch_mnist import MNIST
# from datasets.fetch_cifar10 import CIFAR10
from datasets.fetch_mnist_cnn import MNIST
# from datasets import fetch_mnist_cnn 
from dlgrad.tensor import Tensor
from dlgrad.mlp import MLP
# from datasets.fetch_mnist import MNIST
from dlgrad.afu import ReLU
from nn.training import train, test, plot_metrics 
from dlgrad.graph import save_graph
from dlgrad import optim
import cProfile
import pstats
import time

'''
starting forward

Doing conv for (4, 3, 32, 32) with 6 filters
first conv (4, 6, 28, 28)
relu (4, 6, 28, 28)
first pool (4, 6, 14, 14)

Doing conv for (4, 6, 14, 14) with 16 filters
second conv (4, 16, 10, 10)
second relu (4, 16, 10, 10)
second pool (4, 16, 5, 5)

flatten (4, 400)

    Linear
    data (4, 400)
    weight (120, 400)
    bias (1, 120)
    matmul output (4, 120)
    add output (4, 120)

first fc1 (4, 120)
first relu (4, 120)
    Linear
    data (4, 120)
    weight (84, 120)
    bias (1, 84)
    matmul output (4, 84)
    add output (4, 84)
second fc2 (4, 84)
second relu (4, 84)
    Linear
    data (4, 84)
    weight (10, 84)
    bias (1, 10)
    matmul output (4, 10)
    add output (4, 10)
third fc3 (4, 10)

'''

# class Net:
#     def __init__(self) -> None:
#         self.fc1 = MLP(28*28, 64, bias=True)
#         self.fc3 = MLP(64, 10, bias=True)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = ReLU(x)
#         x = self.fc3(x)
#         return x


# class Net:
#     def __init__(self):
#         self.conv1 = Conv2d(3, 6, 5)
#         self.pool1 = MaxPool2d(2, 2)
#         self.conv2 = Conv2d(6, 16, 5) 
#         self.pool2 = MaxPool2d(2, 2)
#         self.fc1 = MLP(16 * 5 * 5, 120, bias=True)
#         # self.fc1 = MLP(16 * 10 * 10, 120, bias=True)
#         self.fc2 = MLP(120, 84, bias=True)
#         self.fc3 = MLP(84, 10, bias=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = ReLU(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = ReLU(x)
#         x = self.pool2(x)
#         x = Tensor.flatten(x) # flatten all dimensions except batch
#         x = self.fc1(x)
#         x = ReLU(x)
#         x = self.fc2(x)
#         x = ReLU(x)
#         x = self.fc3(x)
#         return x



# kernprof -l -v --unit 1e-3 test.py


class Net:
    def __init__(self):
        self.conv1 = Conv2d(1, 6, 5)
        self.pool1 = MaxPool2d(2, 2)
        # self.conv2 = Conv2d(6, 16, 5) 
        # self.pool2 = MaxPool2d(2, 2)
        # self.fc1 = MLP(338, 20, bias=True)
        # self.fc1 = MLP(400, 120, bias=True)
        # self.fc1 = MLP(256, 20, bias=True)
        
        self.fc1 = MLP(864, 20, bias=True)
        
        # self.fc1 = MLP(16 * 10 * 10, 120, bias=True)
        # self.fc2 = MLP(120, 84, bias=True)
        # self.fc3 = MLP(84, 10, bias=True)
        self.fc3 = MLP(20, 10, bias=True)
    @profile
    def forward(self, x):
        x = self.conv1(x)
        # print(f"first conv {x.shape}")
        x = ReLU(x)
        x = self.pool1(x)
        # x = self.conv2(x)
        # x = ReLU(x)
        # x = self.pool2(x)
        x = Tensor.flatten(x) # flatten all dimensions except batch
        # print(f"flatten {x.shape}")
        x = self.fc1(x)
        x = ReLU(x)
        # x = self.fc2(x)
        # x = ReLU(x)
        x = self.fc3(x)
        return x


# class Net:
#     def __init__(self) -> None:
#         self.fc1 = MLP(28*28, 64, bias=True)
#         self.fc3 = MLP(64, 10, bias=True)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = ReLU(x)
#         x = self.fc3(x)
#         return x

def main():
    epochs = 1 
    BS = 32 
    lr = 1e-3
    
    net = Net()

    start_time = time.perf_counter()
    optimizer = optim.SGD(net, lr)
    
    # cifar_dataset = CIFAR10()
    cifar_dataset = MNIST()
    # x_train.shape (60000, 784)
    # cnn x_train.shape (32000, 3, 32, 32)
    x_train, y_train = cifar_dataset.get_train_data()
    # print(f"x_train {x_train.shape}")
    
    start_time = time.perf_counter()
    for epoch in range(epochs):
        print(f"epoch {epoch+1}")
        cifar_dataset.reset_idx()
        train(net, cifar_dataset, x_train, y_train, BS, optimizer)
    #     with cProfile.Profile() as pr:
    #         train(net, cifar_dataset, x_train, y_train, BS, optimizer, lr)

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    # stats.dump_stats(filename='see.prof')
    
    end_time = time.perf_counter()
    dot_time = end_time - start_time
    print(f"time = {dot_time}")

    # save_graph()

    # x_test, y_test = mnist_dataset.get_test_data()
    # test(net, x_test, y_test)

if __name__ == '__main__':
    main()

