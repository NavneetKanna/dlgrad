import sys
import os
os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad/")
# os.chdir(r"C:/Users/navne/Documents/vs_code/dlgrad")
sys.path.append(os.getcwd())
from dlgrad.mlp import MLP
from dlgrad.conv import Conv2d, MaxPool2d
# from datasets.fetch_mnist import MNIST
from datasets.fetch_fashion_mnist import MNIST
# from datasets.fetch_mnist_cnn import MNIST
from dlgrad.tensor import Tensor
from dlgrad.mlp import MLP
from dlgrad.afu import ReLU
from nn.training import train, test, plot_metrics 
from dlgrad.graph import save_graph
from dlgrad import optim
import time

# kernprof -l -v --unit 1e-3 test.py

class Net:
    def __init__(self):
        self.conv1 = Conv2d(1, 5, 3)
        self.pool1 = MaxPool2d(2, 2)
        self.fc1 = MLP(845, 40, bias=True)
        self.fc3 = MLP(40, 10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = ReLU(x)
        x = self.pool1(x)
        x = Tensor.flatten(x) # flatten all dimensions except batch
        # print(f"flatten {x.shape}")
        x = self.fc1(x)
        x = ReLU(x)
        x = self.fc3(x)
        return x

def main():
    epochs =1 
    BS = 32 
    lr = 1e-3 
    
    net = Net()

    start_time = time.perf_counter()
    optimizer = optim.SGD(net, lr)
    
    fashion_mnist_dataset = MNIST()
    x_train, y_train = fashion_mnist_dataset.get_train_data()
    
    start_time = time.perf_counter()
    for epoch in range(epochs):
            print(f"epoch {epoch+1}")
            fashion_mnist_dataset.reset_idx()
            train(net, fashion_mnist_dataset, x_train, y_train, BS, optimizer)
    
    end_time = time.perf_counter()
    dot_time = end_time - start_time
    print(f"time = {dot_time}")

if __name__ == '__main__':
    main()

