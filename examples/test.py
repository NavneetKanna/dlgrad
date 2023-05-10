import sys
import os
os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/")
sys.path.append(os.getcwd())
from dlgrad.mlp import MLP
from datasets.fetch_mnist import MNIST 
from dlgrad.afu import ReLU
from nn.training import train, test, plot_metrics 
from dlgrad.graph import display_graph

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

# TODO: remove flag, mnist_dataset
def main():
    epochs = 5 
    BS = 128
    lr = 1e-3
    acc_graph, loss_graph = [], []
    
    net = Net()

    for epoch in range(epochs):
        print(f"epoch {epoch+1}")
        
        mnist_dataset = MNIST()
        x_train, y_train = mnist_dataset.get_train_data()

        acc, loss = train(net, mnist_dataset, x_train, y_train, BS, lr, metrics=True)
        acc_graph.append(acc)
        loss_graph.append(loss)

    plot_metrics(acc_graph, loss_graph)
    display_graph()

    x_test, y_test = mnist_dataset.get_test_data()
    test(net, x_test, y_test)


if __name__ == '__main__':
    main()
