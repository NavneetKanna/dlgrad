
from dlgrad.tensor import Tensor
import torch

a = Tensor.rand((4, 2, 3), requires_grad=True)
b = Tensor.rand((1, 3), requires_grad=True)
ta = torch.tensor(a.numpy(), requires_grad=True)
tb = torch.tensor(b.numpy(), requires_grad=True)

c = a+b
c.sum().backward()

tc = ta+tb
tc.sum().backward()

print(a.grad.numpy())
print(b.grad.numpy())
print("--")
print(ta.grad)
print(tb.grad)