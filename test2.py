import numpy as np
import torch

from dlgrad import Tensor

na = np.random.uniform(size=(2, 3)).astype(np.float32)
nb = np.random.uniform(size=(3, 2)).astype(np.float32)

a = Tensor(na, requires_grad=True)
b = Tensor(nb, requires_grad=True)
c=a@b
c.sum().backward()

ta = torch.tensor(na, requires_grad=True)
tb = torch.tensor(nb, requires_grad=True)
tc=ta@tb
tc.sum().backward()

print("dlgrad output")
print(c.numpy())
print("torch output")
print(tc)
print("dlgrad grad")
print(a.grad.numpy())
print(b.grad.numpy())
print("torch grad")
print(ta.grad)
print(tb.grad)

