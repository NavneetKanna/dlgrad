from dlgrad.tensor import Tensor

a = Tensor.rand((4, 3, 2), requires_grad=True)

b = a.sum(dim=1)
# print(a.numpy())
# print()
print(b.numpy())

import torch

ta = torch.tensor(a.numpy())

tb = ta.sum(dim=1)

print(tb)
