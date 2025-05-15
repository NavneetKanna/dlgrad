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
Tensor.full((2, 3), fill_value=4) # Creates a tensor filled with 4
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

With the tensors created, any of the following ops can be performed on them

| Ops |
| :---: |
| add |
| sub |
| mul |
| div |
| matmul |
| transpose |
| sum |
| relu |
| linear |
| max |
| exp |
| log |
| sqrt |
| log_softmax |
| cross_entropy_loss |

For ops like transpose or matmul, you can use the symbols like ```T```, ```@```. Other than these, dlgrad also supports the following operations


| Ops | Left operand | Right operand |
| :---: | :---: | :---: |
| > (greater) | Tensor | Scalar |
| ** (power) | Tensor | Scalar |
| - (negate) | Tensor | - |