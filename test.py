import time

from dlgrad.device import Device
from dlgrad.tensor import Tensor


sh = (1000, 1000)
s = time.perf_counter()
a = Tensor.rand(sh)
e = time.perf_counter()
print(f"{e-s}s")


import torch


s = time.perf_counter()
a = torch.rand(sh, device="cpu")
e = time.perf_counter()
print(f"{e-s:f}s")

from tinygrad import Tensor

s = time.perf_counter()
a = Tensor.rand(sh, device="clang").realize()
e = time.perf_counter()
print(f"{e-s:f}s")