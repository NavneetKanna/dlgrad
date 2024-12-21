from dlgrad.tensor import Tensor
import time

sh = (4, 3, 2)
dim = 2
a = Tensor.rand(sh, requires_grad=True)

s = time.perf_counter()
b = a.sum(dim)
e = time.perf_counter()
print(f"{e-s:.4f}s")
# print(a.numpy())
# print()
print(b.numpy())
print("--")

import torch
torch.set_num_threads(1)
ta = torch.tensor(a.numpy(), device="cpu")
s = time.perf_counter()
tb = ta.sum(dim=dim)
e = time.perf_counter()
print(f"{e-s:.4f}s")

print(tb)

# import tinygrad.tensor as ti

# ta = ti.Tensor(a.numpy(), device="clang")
# s = time.perf_counter()
# tb = ta.sum(axis=dim).realize()
# e = time.perf_counter()
# print(f"{e-s:.4f}s")
