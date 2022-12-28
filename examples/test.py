import sys
import os
# os.chdir(r"C:\\Users\\navne\Documents\\vs_code\\dlgrad_main")
os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/")
sys.path.append(os.getcwd())
from dlgrad.tensor import Tensor
from dlgrad.nn import MLP
from datasets.fetch_mnist import fetch_mnist
from dlgrad.afu import ReLU, softmax 
from nnn.training import train, acc_graph, loss_graph
from dlgrad.graph import display_graph 
import numpy as np
import matplotlib.pyplot as plt

class Net:
    def __init__(self) -> None:
        # self.x_train, self.y_train, self.x_test, self.y_test = fetch_mnist()
        self.batch_size = 32

        self.fc1 = MLP(28*28, 64, bias=True)
        self.fc2 = MLP(64, 64, bias=True)
        self.fc3 = MLP(64, 10, bias=True)

    def forward(self, x_train, flag):
        x = self.fc1(x_train, flag)
        x = ReLU(x, flag)
        x = self.fc2(x, flag)
        x = ReLU(x, flag)
        x = self.fc3(x, flag)
        # x = softmax(x)
        return x

def main():
    epochs = 5 
    x_train, y_train, x_test, y_test = fetch_mnist()
    BS = 32
    flag = True
    net = Net()

    for epoch in range(epochs):
        print(f"epoch {epoch}")
        idx = 0
        for _ in range(int(x_train.shape[0]/BS)):
            X_train = x_train[idx:idx+BS]
            Y_train = y_train[idx:idx+BS]
            idx += BS
            train(net, X_train, Y_train, flag, epoch)
            flag = False
            for parameters in Tensor.save_for_backward:
                parameters.grad = np.zeros(parameters.shape) 

            Tensor.save_for_backward.clear()

    figure, axis = plt.subplots(2, 2)
  
    # axis[0, 0].plot(range(epochs), loss_graph)
    axis[0, 0].plot(loss_graph)
    axis[0, 0].set_title("Loss")
    
    # axis[1, 0].plot(range(epochs), acc_graph)
    axis[1, 0].plot(acc_graph)
    axis[1, 0].set_title("Accuracy")
    axis[1, 0].set_xlabel("Epochs")
  
    plt.close(figure)
    figure.savefig("Metrics.png")

if __name__ == '__main__':
    main()
