import time
from dlgrad.tensor import Tensor
import torch
from tinygrad import tensor

sh = (784, 64)

for i in range(3):
    print("----")
    s = time.perf_counter()
    a = Tensor.rand(sh)
    e = time.perf_counter()
    print(f"{e-s:3f}s")

    s = time.perf_counter()
    a = torch.rand(sh, device="cpu")
    e = time.perf_counter()
    print(f"{e-s:3f}s")

    s = time.perf_counter()
    a = tensor.Tensor.rand(sh, device="clang").realize()
    e = time.perf_counter()
    print(f"{e-s:3f}s")
