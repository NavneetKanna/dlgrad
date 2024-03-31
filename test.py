import random 
import ctypes
import numpy as np
import time
import torch
from tinygrad import Tensor
import dlgrad
from array import array

shape = (7000, 7000)


def to_numpy(fa, l, s):
    sd = ctypes.addressof(fa) + 0 * ctypes.sizeof(ctypes.c_float)
    ptr = (ctypes.c_float * l).from_address(sd)
    data = np.frombuffer(ptr, count=l, dtype=np.float32).reshape(s)
    print(data.shape)

def gf():
    return random.uniform(0, 1)

def new2():
    s = time.perf_counter()
    a = dlgrad.Tensor.rand(shape)
    e = time.perf_counter()
    print(f"new2 {e-s:.4f}s")

def new():
    s = time.perf_counter()
    a = (ctypes.c_float * (shape[0]*shape[1]))()
    for i in range(shape[0]*shape[1]):
        a[i] = random.random()
    e = time.perf_counter()
    print(f"new {e-s:.4f}s")
    # to_numpy(a, shape[0]*shape[1], shape)

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

new2()
new()
num()
to()
ti()