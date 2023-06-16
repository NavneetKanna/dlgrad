# import sys
# import os
# os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad/")
# sys.path.append(os.getcwd())

import unittest
import torch 
from torch import nn
import numpy as np
import ctypes






im2col_lib = ctypes.CDLL('/mnt/c/Users/navne/Documents/vs_code/dlgrad/so/im2col.so')
col2im_lib = ctypes.CDLL('/mnt/c/Users/navne/Documents/vs_code/dlgrad/so/col2im.so')

# Define the function prototype
im2col_c_func = im2col_lib.im2col_c
im2col_c_func.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int
)

col2im_c_func = col2im_lib.col2im_c
col2im_c_func.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int
)

INP_W = 4
INP_H = 4

BS = 2

in_channels = 2 
out_channels = 2 
kernal_size = 2 
torch_conv_weight = None

conv_upstream_gradient = None

def conv_backward_hook(module, grad_input, grad_output):
    global conv_upstream_gradient
    # Access the upstream gradient
    conv_upstream_gradient = grad_output[0]

# Same implementation as in conv.py
def conv(data,  stride=1, padding=0):
    new_w = (data.shape[2]-kernal_size) // stride + 1
    new_h = (data.shape[3]-kernal_size) // stride + 1 
    X_col = np.zeros((data.shape[0]*new_w*new_h, data.shape[1]*kernal_size*kernal_size), dtype=np.float32)
    weight = torch_conv_weight.astype(np.float32)
    out_w = ((data.shape[-2] - kernal_size + 2*padding)//stride) + 1 
    out_h = ((data.shape[-1] - kernal_size + 2*padding)//stride) + 1 
    conv_out = np.zeros((data.shape[0], out_channels, out_w, out_h), dtype=np.float32)
    im2col_c_func(data, X_col, data.shape[0], data.shape[1], data.shape[2], data.shape[3], kernal_size, stride)
    w_col = weight.reshape((out_channels, -1))
    out =  X_col @ w_col.T 
    conv_out = np.array(np.hsplit(out.T, data.shape[0])).reshape((data.shape[0], out_channels, out_h, out_w))
    cache = data, X_col, w_col
    return conv_out, cache

# Same implementation as in conv.py
def backward(data, cache, stride=1):
    X, X_col, w_col = cache
    m, _, _, _ = X.shape
    conv_out_grad = conv_upstream_gradient.detach().numpy() 
    dout = conv_out_grad.reshape(conv_out_grad.shape[0] * conv_out_grad.shape[1], conv_out_grad.shape[2] * conv_out_grad.shape[3])
    dout = np.array(np.vsplit(dout, m))
    dout = np.concatenate(dout, axis=-1).T
    dX_col = dout @ w_col
    dw_col = (X_col.T @ dout).T
    data_grad = np.zeros(data.shape, dtype=np.float32)
    col2im_c_func(data_grad, dX_col, data.shape[0], data.shape[1], data.shape[2], data.shape[3], kernal_size, stride)
    weight_grad = dw_col.reshape((dw_col.shape[0], in_channels, kernal_size, kernal_size))
    return data_grad, weight_grad


class TestConv(unittest.TestCase):
    def test_conv_forward(self):
        # print("Testing Conv forward and backward with same weight and upstream gradient")
        global torch_conv_weight
        self.torch_conv = nn.Conv2d(in_channels, out_channels, kernal_size, bias=False)

        self.torch_conv.register_full_backward_hook(conv_backward_hook)
        for p in self.torch_conv.parameters():
            torch_conv_weight = p.detach().numpy()

        self.inp = torch.randn(BS, in_channels, INP_W, INP_H).requires_grad_()

        self.torch_conv_output = self.torch_conv(self.inp)
        self.my_output, self.cache = conv(self.inp.detach().numpy())

        np.testing.assert_allclose(self.torch_conv_output.detach().numpy().round(3), self.my_output.round(3), rtol=0, atol=10**(-2))

        upstream_gradients = torch.randn_like(self.torch_conv_output)

        # Backward pass 
        self.torch_conv_output.backward(gradient=upstream_gradients) 
        my_data_grad, my_weight_grad = backward(self.inp.detach().numpy(), self.cache)

        np.testing.assert_allclose(self.inp.grad.detach().numpy().round(3), my_data_grad.round(3), rtol=0, atol=10**(-2))
        np.testing.assert_allclose(self.torch_conv.weight.grad.detach().numpy().round(3), my_weight_grad.round(3), rtol=0, atol=10**(-2))
