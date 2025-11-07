## dlgrad

This guide will explain all the things dlgrad can do.

### Tensors

Tensors are multidimensional arrays based on which all operations are performed on. It is similar to pytorch tensors.

The tensor class can be imported like so

```python
from dlgrad import Tensor
```

Tensors can be created from numpy arrays

```python
Tensor(np.array([1, 2, 3], dtype=np.float32))
```

or from one of the data creation methods

```python
Tensor.uniform((2, 3), low=-1, high=1) # Creates a tensor filled with random numbers from a uniform distribution on the interval [low, high)
Tensor.rand((2, 3)) # Creates a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
Tensor.full((2, 3), fill_value=4) # Creates a tensor from a uniform distribution filled with 4
Tensor.ones_like((2, 3)) # Creates a tensor filled with 1
Tensor.zeros_like((2, 3)) # Creates a tensor filled with 0
```

All of the above functions take ```dtype``` and ```device``` as arguments, the supported dtypes are

| Dtype |
| :---: |
| float32 |

and the supported devices are

| Device |
| :---: |
| cpu |
| metal |

With the tensors created, any of the following ops can be performed on them, the table also shows which runtime supports which ops and which dimensions

| Ops | CPU | Metal | Dim |
| :---: | :---: | :---: | :---: |
| add | &check; | &check; | 1D-4D |
| sub | &check; | &check; | 1D-4D |
| mul | &check; | &check; | 1D-4D |
| div | &check; | &check; | 1D-4D |
| matmul | &check; | &check; | 2D |
| transpose | &check; | &check; | 2D |
| sum | &check; | &check; | 1D-4D |
| relu | &check; | &check; | 1D-4D |
| leaky_relu | &check; | &check; | 1D-4D |
| tanh | &check; | &check; | 1D-4D |
| linear | &check; | - |
| max | &check; | &check; | 1D-4D |
| exp | &check; | &check; | 1D-4D |
| log | &check; | &check; | 1D-4D |
| sqrt | &check; | &check; | 1D-4D |
| clamp | &check; | &cross; | 1D-4D |
| log_softmax | &check; |  - |
| cross_entropy_loss | &check; | - |
| argmax | &check; | &cross; | 2D |


For ops like transpose or matmul, you can use the symbols like ```T```, ```@```. Other than these, dlgrad also supports the following operations


| Ops | Left operand | Right operand | CPU | Metal
| :---: | :---: | :---: |:---: | :---: |
| > (greater) | Tensor | Scalar | &check| &cross;
| ** (power) | Tensor | Scalar | &check | &check; 2D, 3D |
| - (negate) | Tensor | - | &check | &check; 2D, 3D |

### Models

You can create neural networks models like so

```python
class Model:
    def __init__(self):
        self.layers = [
            nn.Linear(in_dim, HS, bias=True),
            Tensor.relu,
            nn.Linear(HS, ncls, bias=True)
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)
```

then we create an instance of the class and we can begin training.

```python
model = Model()
```

### Training

dlgrad supports the following loss functions

| Loss Functions |
| :---: |
| Cross-Entropy Loss |

You can use it like so

```python
model_output.cross_entropy_loss(target=x_train_labels[s:h])
```

The following optimizers are supported

| Optimizers |
| :---: |
| SGD |
| Adam |

and we can use them like this

```python
opt = nn.optim.Adam(params=nn.utils.get_parameters(model), lr=1e-3)
```

We can use ```nn.utils.get_parameters()``` to get all the trainable parameters of the defined model automatically.

Finally, we can call the ```backward()``` function to propagate the loss and ```opt.step()``` to update the weights of the network.

### Indexing

dlgrad only supports slicing indexing.

```python
a = Tensor.rand((3, 2, 3))
a[0:1]
a[1:2]
```

### Printing

To print a tensor, you need to call the ```.numpy()``` method

```python
a = Tensor.rand((2, 3))
print(a.numpy())
```

### Properties

Every tensor has got the following properties

- numel
- shape
- stride
- ndim
- dtype
- device


### Dataloaders

dlgrads comes with ```mnsit``` loader, to use them just import call the function

```python
from dlgrad.nn.datasets import mnist

# (60000, 784), (60000, 1), (10000, 784), (10000, 1)
x_train_images, x_train_labels, x_test_images, x_test_labels = mnist(device="metal")

```
