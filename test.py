import time

from dlgrad.device import Device
from dlgrad.tensor import Tensor


# a = Tensor.rand((2, 3))
# s = time.perf_counter()
a = Tensor(1)
print(a)
# e = time.perf_counter()
# print(f"{e-s}s")


# import torch


# s = time.perf_counter()
# a = torch.tensor(1, device="cpu")
# e = time.perf_counter()
# print(f"{e-s:f}s")

# from tinygrad import Tensor

# s = time.perf_counter()
# a = Tensor(1, device="clang")
# e = time.perf_counter()
# print(f"{e-s:f}s")