from dlgrad.device import Device
from dlgrad.tensor import Tensor

import time

s = time.perf_counter()
a = Tensor(1)
e = time.perf_counter()
print(f"{e-s}s")


import torch

s = time.perf_counter()
a = torch.tensor(1, device="cpu")
e = time.perf_counter()
print(f"{e-s:f}s")

from tinygrad import Tensor

s = time.perf_counter()
a = Tensor(1, device="clang")
e = time.perf_counter()
print(f"{e-s:f}s")