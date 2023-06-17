--------------------------------------------------------------------

This work is inspired by myself, [Andrej Karpathy micrograd](https://github.com/karpathy/micrograd) and [George Hotz tinygrad](https://github.com/geohot/tinygrad).

The purpose of this porject is to increase my knowledge in deep learning and to understand how everything works under the hood.  


--------------------------------------------------------------------
## CNN 
```python

from datasets.fetch_cifar10 import CIFAR10
from dlgrad.mlp import MLP
from dlgrad.conv import Conv2d, MaxPool2d
from dlgrad.tensor import Tensor
from dlgrad import optim
from nn.training import train, test, plot_metrics 

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
    epochs = 3 
    BS = 64 
    lr = 1e-3
    
    net = Net()

    start_time = time.perf_counter()
    optimizer = optim.SGD(net, lr)
    
    cifar_dataset = CIFAR10()
    x_train, y_train = cifar_dataset.get_train_data()
    
    for epoch in range(epochs):
        print(f"epoch {epoch+1}")
        fashion_mnist_dataset.reset_idx()
        train(net, cifar_dataset, x_train, y_train, BS, optimizer)

```



## Simple Network (ANN)
```python
from dlgrad.mlp import MLP
from datasets.fetch_mnist import MNIST 
from nn.training import train, test, plot_metrics 
from dlgrad.graph import display_graph
from dlgrad import optim

class Net:
    def __init__(self) -> None:
        self.fc1 = MLP(28*28, 64, bias=True)
        self.fc3 = MLP(64, 10, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = x.ReLU()
        x = self.fc3(x)
        return x

def main():
    epochs = 3 
    BS = 128
    lr = 1e-3
    
    net = Net()

    optimizer = optim.SGD(net, lr)
    
    mnist_dataset = MNIST()
    x_train, y_train = mnist_dataset.get_train_data()
    
    for epoch in range(epochs):
        print(f"epoch {epoch+1}")
        mnist_dataset.reset_idx()
        train(net, mnist_dataset, x_train, y_train, BS, optimizer, lr)

    save_graph()

    x_test, y_test = mnist_dataset.get_test_data()
    test(net, x_test, y_test)
```


--------------------------------------------------------------------
## Testing
To run the tests go to main dir (dlgrad/) and run the command
```shell

python -m unittest discover 

```

--------------------------------------------------------------------
## Computational Graph for the ANN model
<p align="center">
  <img src="https://github.com/NavneetKanna/dlgrad/blob/main/Images/graph.png?raw=true">
</p>


