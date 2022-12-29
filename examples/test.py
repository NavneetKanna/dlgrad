import sys
import os
# os.chdir(r"C:\\Users\\navne\Documents\\vs_code\\dlgrad_main")
os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/")
sys.path.append(os.getcwd())
from dlgrad.tensor import Tensor
from dlgrad.nn import MLP
from datasets.fetch_mnist import MNIST 
from dlgrad.afu import ReLU, softmax 
from nnn.training import train, test, plot_metrics 
from dlgrad.graph import display_graph 
import numpy as np
import matplotlib.pyplot as plt

class Net:
    def __init__(self) -> None:
        self.fc1 = MLP(28*28, 64, bias=True)
        self.fc2 = MLP(64, 64, bias=True)
        self.fc3 = MLP(64, 10, bias=True)

    def forward(self, x_train, flag=False):
        x = self.fc1(x_train, flag)
        x = ReLU(x, flag)
        x = self.fc2(x, flag)
        x = ReLU(x, flag)
        x = self.fc3(x, flag)
        # x = softmax(x)
        return x

# TODO: remove flag
def main():
    epochs = 5
    BS = 128
    lr=1e-3
    flag = True
    
    net = Net()

    for epoch in range(epochs):
        print(f"epoch {epoch+1}")
        
        mnist_dataset = MNIST()
        x_train, y_train = mnist_dataset.get_train_data()
        steps = x_train.shape[0]//BS

        train(net, mnist_dataset, x_train, y_train, steps, BS, flag, lr)

        flag = False
    print("plotting metrics")   
    plot_metrics()
    print("finished")

    x_test, y_test = mnist_dataset.get_test_data()
    test(net, x_test, y_test)


if __name__ == '__main__':
    main()
