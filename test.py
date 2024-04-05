import random 
import ctypes
import numpy as np
import time
import torch
from tinygrad import Tensor
import dlgrad

shape = (100, 100)


def to_numpy(fa, l, s):
    sd = ctypes.addressof(fa) + 0 * ctypes.sizeof(ctypes.c_float)
    ptr = (ctypes.c_float * l).from_address(sd)
    data = np.frombuffer(ptr, count=l, dtype=np.float32).reshape(s)
    print(data.shape)

def dl():
    s = time.perf_counter()
    a = dlgrad.Tensor.rand(shape)
    e = time.perf_counter()
    print(f"dl {e-s:.4f}s")

def num():
    s = time.perf_counter()
    a = np.random.randn(*shape)
    e = time.perf_counter()
    print(f"numpy {e-s:.4f}s")

def to():
    s = time.perf_counter()
    a = torch.rand(shape)
    e = time.perf_counter()
    print(f"torch {e-s:.4f}s")

def ti():
    s = time.perf_counter()
    a = Tensor.rand(shape).numpy()
    e = time.perf_counter()
    print(f"tinygrad {e-s:.4f}s")

dl()
num()
to()
ti()