import torch

from dlgrad.tensor import Tensor

a = Tensor.rand((2, 3, 3), requires_grad=True)
ta = torch.tensor(a.numpy(), requires_grad=True)

c = a.sum((0, 1))
tc = ta.sum((0, 1))

print(c.numpy())
print(tc.numpy())
# b = Tensor.rand((1, 3, 3), requires_grad=True)
# c=a+b
# c.sum().backward()

# print(a.grad.numpy())
# print(b.grad.numpy())
