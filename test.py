from dlgrad.tensor import Tensor
import torch

# a = Tensor.rand((1, 3))
# b = Tensor.rand((2, 3))
# c = a-b

# print("\n-----")
# print(a.numpy())
# print(b.numpy())
# print(c.numpy())

import numpy as np

a = np.random.uniform(size=(3, 2)).astype(np.float32)
b = np.random.uniform(size=(4, 3, 2)).astype(np.float32)
print(a)
print(b)

da = Tensor(a)
db = Tensor(b)

print((da+db).numpy())

ta = torch.tensor(a)
tb = torch.tensor(b)
print((ta+tb).stride())
print(ta+tb)