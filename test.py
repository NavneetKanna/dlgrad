# import time

from dlgrad.tensor import Tensor
import numpy as np


a = np.random.rand(2, 3).astype(np.float32)
print(a)
print("--")
d = Tensor(a)

print(d.numpy())