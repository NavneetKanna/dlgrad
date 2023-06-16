import unittest
import torch
from torch import nn
import numpy as np
import ctypes

maxpool_forward_lib = ctypes.CDLL('/mnt/c/Users/navne/Documents/vs_code/dlgrad/so/maxpool_forward.so')

maxpool_forward_c_func = maxpool_forward_lib.maxpool_forward_c
maxpool_forward_c_func.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), 
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int
)

maxpool_backward = ctypes.CDLL('/mnt/c/Users/navne/Documents/vs_code/dlgrad/so/maxpool_backward.so')

maxpool_backward_c_func = maxpool_backward.maxpool_backward_c
maxpool_backward_c_func.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), 
    ctypes.c_int 
)

maxpool_upstream_gradient = None

MAXPOOL_SIZE = 2
MAXPOOL_STRIDE = 2
BS = 2
C = 3
INP_W = 28
INP_H = 28

def maxpool(data, maxsize=2, maxstride=2):
    N, C, W, H = data.shape
    new_w = ((W-maxsize) // maxstride) + 1
    new_h = ((H-maxsize) // maxstride) + 1
    out = np.zeros((N, C, new_w, new_h), dtype=np.float32)
    max_idx = np.zeros((N, C, new_w, new_h), dtype=np.int32)
    maxpool_forward_c_func(
        data, 
        out, 
        max_idx, 
        data.shape[0], 
        data.shape[1], 
        data.shape[2], 
        data.shape[3], 
        maxsize, 
        maxstride
    ) 
    return out, max_idx


def maxpool_backward(data, max_idx):
    data_grad = np.zeros(data.shape, dtype=np.float32)
    out_grad = maxpool_upstream_gradient.detach().numpy() 
    maxpool_backward_c_func(
        data, 
        data_grad, 
        out_grad, 
        max_idx, 
        data.shape[0], 
        data.shape[1], 
        data.shape[2], 
        data.shape[3], 
        MAXPOOL_SIZE, 
        MAXPOOL_STRIDE
    )
    return data_grad


def maxpool_backward_hook(module, grad_input, grad_output):
    global maxpool_upstream_gradient
    # Access the upstream gradient
    maxpool_upstream_gradient = grad_output[0]


class TestMaxpool(unittest.TestCase):
    def test_maxpool(self):
        torch_maxpool = nn.MaxPool2d(MAXPOOL_SIZE, MAXPOOL_STRIDE)
        torch_maxpool.register_full_backward_hook(maxpool_backward_hook)

        inp = torch.randn(BS, C, INP_W, INP_H).requires_grad_()

        torch_pool_output = torch_maxpool(inp)
        my_maxpool_output, max_idx = maxpool(inp.detach().numpy())

        np.testing.assert_allclose(torch_pool_output.detach().numpy().round(3), my_maxpool_output.round(3), rtol=0, atol=10**(-2))

        pool_upstream_gradients = torch.randn_like(torch_pool_output)
        torch_pool_output.backward(gradient=pool_upstream_gradients) 
        my_pool_data_grad = maxpool_backward(inp.detach().numpy(), max_idx)

        np.testing.assert_allclose(inp.grad.detach().numpy().round(3), my_pool_data_grad.round(3), rtol=0, atol=10**(-2))
