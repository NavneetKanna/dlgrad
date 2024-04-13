# import ctypes
import numpy as np
import time
import torch
from tinygrad import Tensor
import dlgrad
import mlx.core as mx

shape = (5000, 5000)

# def to_numpy(fa, l, s):
#     sd = ctypes.addressof(fa) + 0 * ctypes.sizeof(ctypes.c_float)
#     ptr = (ctypes.c_float * l).from_address(sd)
#     data = np.frombuffer(ptr, count=l, dtype=np.float32).reshape(s)
#     print(data.shape)

print(f"---- create rand buffer {shape} on cpu ----")
def create_rand_dl():
    s = time.perf_counter()
    _ = dlgrad.Tensor.rand(shape, device='cpu')
    e = time.perf_counter()
    print(f"dlgrad {e-s:.4f}s")

def create_rand_num():
    s = time.perf_counter()
    _ = np.random.randn(*shape)
    e = time.perf_counter()
    print(f"numpy {e-s:.4f}s")

def create_rand_to():
    s = time.perf_counter()
    _ = torch.rand(shape, device='cpu')
    e = time.perf_counter()
    print(f"torch {e-s:.4f}s")

def create_rand_ti():
    s = time.perf_counter()
    _ = Tensor.rand(shape, device='clang').numpy()
    e = time.perf_counter()
    print(f"tinygrad {e-s:.4f}s")

def create_rand_mlx():
    s = time.perf_counter()
    _ = mx.eval(mx.random.uniform(shape=shape, stream=mx.cpu))
    e = time.perf_counter()
    print(f"mlx {e-s:.4f}s")

create_rand_dl()
create_rand_num()
create_rand_to()
create_rand_ti()
create_rand_mlx()

print(f"---- create rand buffer {shape} on cpu, but calling dlgrad the second time ----")
create_rand_dl()
create_rand_num()
create_rand_to()
create_rand_ti()
create_rand_mlx()

print(f"---- add {shape} on cpu ----")
def dl_add():
    a = dlgrad.Tensor.rand(shape, device='cpu')
    b = dlgrad.Tensor.rand(shape, device='cpu')
    s = time.perf_counter()
    _ = dlgrad.Tensor.add(a, b)
    e = time.perf_counter()
    print(f"dlgrad {e-s:.4f}s")

def num_add():
    a = np.random.rand(*shape)
    b = np.random.rand(*shape)
    s = time.perf_counter()
    _ = a+b
    e = time.perf_counter()
    print(f"numpy {e-s:.4f}s")

def to_add():
    a = torch.rand(shape, device='cpu')
    b = torch.rand(shape, device='cpu')
    s = time.perf_counter()
    _ = a+b
    e = time.perf_counter()
    print(f"torch {e-s:.4f}s")

def ti_add():
    a = Tensor.rand(shape, device='clang')
    b = Tensor.rand(shape, device='clang')
    s = time.perf_counter()
    _ = Tensor.add(a, b).numpy()
    e = time.perf_counter()
    print(f"tinygrad {e-s:.4f}s")

def mlx_add():
    a = mx.random.uniform(shape=shape, stream=mx.cpu)
    b = mx.random.uniform(shape=shape, stream=mx.cpu)
    s = time.perf_counter()
    _ = mx.eval(a+b)
    e = time.perf_counter()
    print(f"mlx {e-s:.4f}s")

dl_add()
num_add()
to_add()
ti_add()
mlx_add()

print(f"---- add {shape} on cpu, but calling dlgrad the second time ----")
dl_add()
num_add()
to_add()
ti_add()
mlx_add()