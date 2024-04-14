# dlgrad

dlgrad (*D*eep *L*earning auto*grad*): A Lightweight Autograd Engine for Deep Learning

Inspired by Andrej Karpathy's micrograd and George Hotz's tinygrad, dlgrad is my personal exploration into building an Autograd engine from scratch. Its lightweight in design and has PyTorch like API.

## Features

- **CPU and GPU Support**: The library currently supports both CPU and GPU (Metal) backends.

## Internals

You can read my [blog](https://navneetkanna.github.io/blog/2024/02/22/dlgrad-Behind-the-scenes.html) to learn more about how dlgrad operates.

## Examples

```python
from dlgrad.tensor import Tensor

# Create tensors filled with random numbers from a uniform distribution
a = Tensor.rand(10, 10)
b = Tensor.rand(10, 10)
# Since the tensors are c buffers, use numpy to print
print(a.numpy())
print(b.numpy())

c = Tensor.add(a, b)
print(c.numpy())

```


<!-- --------------------------------------------------------------------
## Computational Graph for the ANN model
<p align="center">
  <img src="https://github.com/NavneetKanna/dlgrad/blob/main/Images/graph.png?raw=true">
</p>  -->


