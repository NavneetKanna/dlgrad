import sys
import os
os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad/")
sys.path.append(os.getcwd())
from dlgrad.mlp import MLP
from datasets.fetch_mnist import MNIST
from dlgrad.afu import ReLU
from nn.training import train, test, plot_metrics 
from dlgrad.graph import save_graph
from dlgrad import optim
import cProfile
import pstats
import time

class Net:
    def __init__(self) -> None:
        self.fc1 = MLP(28*28, 64, bias=True)
        self.fc3 = MLP(64, 10, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = ReLU(x)
        x = self.fc3(x)
        return x

def main():
    epochs = 5 
    BS = 64 
    lr = 1e-3
    
    net = Net()

    optimizer = optim.SGD(net, lr)

    start_time = time.perf_counter()
    for epoch in range(epochs):
        print(f"epoch {epoch+1}")
        
        mnist_dataset = MNIST()
        x_train, y_train = mnist_dataset.get_train_data()

        train(net, mnist_dataset, x_train, y_train, BS, optimizer, lr)
        # with cProfile.Profile() as pr:
        #     train(net, mnist_dataset, x_train, y_train, BS, optimizer, lr)
        
        # stats = pstats.Stats(pr)
        # stats.sort_stats(pstats.SortKey.TIME)
        # stats.print_stats()
        # stats.dump_stats(filename='see.prof')
    end_time = time.perf_counter()
    dot_time = end_time - start_time
    print(f"time = {dot_time}")

    # save_graph()

    x_test, y_test = mnist_dataset.get_test_data()
    test(net, x_test, y_test)


if __name__ == '__main__':
    main()
