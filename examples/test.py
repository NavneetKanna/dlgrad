import sys
import os
os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad/")
sys.path.append(os.getcwd())
from dlgrad.mlp import MLP
from datasets.fetch_mnist import MNIST 
from dlgrad.afu import ReLU
from nn.training import train, test, plot_metrics 
from dlgrad.graph import display_graph
from dlgrad import optim

class Net:
    def __init__(self) -> None:
        self.fc1 = MLP(28*28, 64, bias=True)
        # self.fc2 = MLP(64, 64, bias=True)
        self.fc3 = MLP(64, 10, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = ReLU(x)
        # x = self.fc2(x, flag)
        # x = ReLU(x, flag)
        x = self.fc3(x)
        # x = softmax(x)
        return x

def main():
    epochs = 3 
    BS = 128
    lr = 1e-3
    
    net = Net()

    optimizer = optim.SGD(net, lr)

    for epoch in range(epochs):
        print(f"epoch {epoch+1}")
        
        mnist_dataset = MNIST()
        x_train, y_train = mnist_dataset.get_train_data()

        train(net, mnist_dataset, x_train, y_train, BS, optimizer, lr)

    display_graph()

    x_test, y_test = mnist_dataset.get_test_data()
    test(net, x_test, y_test)


if __name__ == '__main__':
    main()
