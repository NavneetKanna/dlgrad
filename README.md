--------------------------------------------------------------------

This work is inspired by myself, [Andrej Karpathy micrograd](https://github.com/karpathy/micrograd) and [George Hotz tinygrad](https://github.com/geohot/tinygrad).

The purpose of this porject is to increase my knowledge in deep learning and to understand how everything works intuitively.  


--------------------------------------------------------------------

## MNIST Example
```python
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
        return x

def main():
    epochs = 5
    BS = 128
    lr = 1e-3
    flag = True
    
    net = Net()

    for epoch in range(epochs):
        mnist_dataset = MNIST()
        x_train, y_train = mnist_dataset.get_train_data()
        steps = x_train.shape[0]//BS

        train(net, mnist_dataset, x_train, y_train, steps, BS, flag, lr)

        flag = False
    
    plot_metrics()

    x_test, y_test = mnist_dataset.get_test_data()
    test(net, x_test, y_test)
```

## Computational Graph
<p>
  <img src="dlgrad/graph.png" width='100' height='100'>
</p>