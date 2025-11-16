## dlgrad

dlgrad is a lightweight deep learning engine built from scratch. It’s designed to be minimal, fast, and fully transparent. It supports two accelerators, CPU and Apple Silicon GPUs. This guide will explain all the things dlgrad can do.

### Tensors

Tensors are multidimensional arrays based on which all operations are performed. It is similar to PyTorch tensors.

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
a = Tensor.rand((2, 3)) # Creates a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
Tensor.uniform((2, 3), low=-1, high=1) # Creates a tensor filled with random numbers from a uniform distribution on the interval [low, high)
Tensor.full((2, 3), fill_value=4) # Creates a tensor from a uniform distribution filled with 4
Tensor.ones_like(a) # Creates a tensor filled with 1
Tensor.zeros_like(a) # Creates a tensor filled with 0
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

| Ops | CPU | Metal | Dim | Notes |
| :---: | :---: | :---: | :---: | :---: |
| add | &check; | &check; | 1D-4D |
| sub | &check; | &check; | 1D-4D |
| mul | &check; | &check; | 1D-4D |
| div | &check; | &check; | 1D-4D |
| matmul | &check; | &check; | 2D |
| transpose | &check; | &check; | 2D |
| sum | &check; | &check; | 1D-4D | Metal supports sum along last dim or full tensor |
| relu | &check; | &check; | 1D-4D |
| sigmoid | &check; | &check; | 1D-4D |
| mean | &check; | &cross; | 1D-4D |
| leaky_relu | &check; | &check; | 1D-4D |
| tanh | &check; | &check; | 1D-4D |
| linear | &check; | - |
| max | &check; | &check; | 1D-4D | Metal supports max along last dim or full tensor |
| exp | &check; | &check; | 1D-4D |
| log | &check; | &check; | 1D-4D |
| sqrt | &check; | &check; | 1D-4D |
| clamp | &check; | &cross; | 1D-4D |
| log_softmax | &check; |  - |
| cross_entropy_loss | &check; | - |
| bce_with_logits_loss | &check; | - |
| argmax | &check; | &cross; | 2D |
| where | &check; | &cross; | 2D |
| squeeze | - | - | - |
| unsqueeze | - | - | - |

For ops like transpose or matmul, you can use shorthand symbols such as ```T```, ```@```. Other than these, dlgrad also supports the following operations

| Ops | Left operand | Right operand | CPU | Metal | Dim |
| :---: | :---: | :---: | :---: | :---: | :---: |
| > (greater) | Tensor | Scalar | &check; | &cross; | 1D-4D |
| ** (power) | Tensor | Scalar | &check; | &check; | 1D-4D |
| - (negate) | Tensor | - | &check; | &check; | 1D-4D |
| == (equate) | Tensor | Tensor | &check; | &cross; | 1D-4D |

### Models

You can create neural network models like so

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

then create an instance of the class and we can begin training.

```python
model = Model()
```

### Training

dlgrad supports the following loss functions

| Loss Functions |
| :---: |
| Cross-Entropy Loss |
| BCEWithLogitsLoss |

The following optimizers are supported

| Optimizers |
| :---: |
| SGD |
| Adam |

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

To print a tensor, you need to call the ```.numpy()``` method which returns a NumPy array

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

dlgrads comes with an ```mnist``` loader

```python
from dlgrad.nn.datasets import mnist

# (60000, 784), (60000, 1), (10000, 784), (10000, 1)
x_train_images, x_train_labels, x_test_images, x_test_labels = mnist(device="metal")
```
These are unnormalized.
