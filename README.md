# dlgrad

dlgrad (*D*eep *L*earning auto*grad*): A Lightweight Autograd Engine for Deep Learning

Inspired by Andrej Karpathy's micrograd and George Hotz's tinygrad, dlgrad is my personal exploration into building an Autograd engine from scratch. Its lightweight in design and has PyTorch like API.

## Features

- **CPU and GPU Support**: The library currently supports CPU backend and GPU support is coming in future.

## Internals

You can read my [blog](https://navneetkanna.github.io/blog/2024/02/22/dlgrad-Behind-the-scenes.html) to learn more about how dlgrad operates.

## Things I'm Working On
- [x] nn.Linear
- [x] Activation functions (relu)
- [ ] Loss functions

## Examples

```python
# Can call with GRAPH=1 to visualise the computational graph
 
from dlgrad.tensor import Tensor

# Create tensors filled with random numbers from a uniform distribution
a = Tensor.rand(2, 3)
b = Tensor.rand(1, 3)
# Since the tensors are c buffers, use numpy to print
a.numpy()
b.numpy()

c = a+b

# Do a backward pass
c.sum().backward()

a.grad.numpy()
b.grad.numpy()

```

## Testing

To run all the tests, run the command from the root dir

```bash
python3 -m unittest discover test
```