import tinygrad.tensor
from dlgrad import Tensor
import numpy as np
import torch
import time
import tinygrad


sh1 = (200, 300)
sh2 = (300, 200)
na = np.random.uniform(size=sh1).astype(np.float32)
nb = np.random.uniform(size=sh2).astype(np.float32)

da = Tensor(na)
db = Tensor(nb)
s = time.perf_counter()
da@db
e = time.perf_counter()
print(f"dlgrad: {e-s:f}s")

ta = torch.tensor(na, device="cpu")
tb = torch.tensor(nb, device="cpu")
s = time.perf_counter()
ta@tb
e = time.perf_counter()
print(f"torch: {e-s:f}s")

tia = tinygrad.tensor.Tensor(na, device="clang")
tib = tinygrad.tensor.Tensor(nb, device="clang")
s = time.perf_counter()
(tia@tib).realize()
e = time.perf_counter()
print(f"tinygrad: {e-s:f}s")

