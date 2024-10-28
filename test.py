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


"""
-OFast
----
0.000189s
0.000265s
0.085911s
----
0.000171s
0.000151s
0.000209s
----
0.000154s
0.000131s
0.000187s


-O3
----
0.000191s
0.000263s
0.085846s
----
0.000171s
0.000152s
0.000210s
----
0.000155s
0.000130s
0.000184s

-O2
----
0.000184s
0.000259s
0.080407s
----
0.000176s
0.000150s
0.000209s
----
0.000151s
0.000129s
0.000185s

"""