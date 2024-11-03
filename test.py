import time
from dlgrad.tensor import Tensor
import torch
from tinygrad import tensor

s1 = (100, 100)
s2 = (100, 100)

a = Tensor.rand(s1)
b = Tensor.rand(s2)

s = time.perf_counter()
a+b
e = time.perf_counter()
print(f"{e-s:f}")

a = torch.rand(s1)
b = torch.rand(s2)

s = time.perf_counter()
a+b
e = time.perf_counter()
print(f"{e-s:f}")

a = tensor.Tensor.rand(s1, device="clang")
b = tensor.Tensor.rand(s2, device="clang")

s = time.perf_counter()
(a+b).realize()
e = time.perf_counter()
print(f"{e-s:f}")
