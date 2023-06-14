import numpy as np
from .tensor import Tensor
from .helper import backward_list
import ctypes

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)
# https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays

im2col_lib = ctypes.CDLL('so/im2col.so')
col2im_lib = ctypes.CDLL('so/col2im.so')

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

maxpool_forward_lib = ctypes.CDLL('so/maxpool_forward.so')

# Define the function prototype
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

maxpool_backward = ctypes.CDLL('so/maxpool_backward.so')

maxpool_backward_c_func = maxpool_backward.maxpool_backward_c
maxpool_backward_c_func.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'), 
    ctypes.c_int 
)

class Conv2d:
    # nn.conv2d(in_channels, out_channels, kernel_size) = (out_channels, in_channels, kernel_size, kernel_size)
    # nn.conv2d(3, 6, 5) = (6, 3, 5, 5)
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        self.in_channels = in_channels # No of Channles
        self.out_channels = out_channels # No of Filters/Kernels
        self.kernal_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.matmul_output = []

        # fan_in = in_channels * (kernel_size[0] * kernel_size[1])

        std = np.sqrt(2.0 / (self.in_channels * self.kernal_size * self.kernal_size))
        self.weight = Tensor(np.random.randn(self.out_channels, self.in_channels, self.kernal_size, self.kernal_size).astype(np.float32) * std, 'Conv Weight')
        # self.weight = Tensor.uniform((self.out_channels, self.in_channels, self.kernal_size, self.kernal_size))
        
    def __call__(self, data: Tensor) -> Tensor: 
        backward_list.extend((data, self.weight))
        self.new_w = ((data.shape[-2] - self.kernal_size + 2*self.padding)//self.stride) + 1 
        self.new_h = ((data.shape[-1] - self.kernal_size + 2*self.padding)//self.stride) + 1 
        self.conv_out = Tensor(np.zeros((data.shape[0], self.out_channels, self.new_w, self.new_h), dtype=np.float32), 'Conv Output')
        data.grad = np.zeros(data.tensor.shape, dtype=np.float32)
        return self.conv(data)

    def conv(self, data) -> Tensor:
        # new_w = (data.shape[2]-self.kernal_size) // self.stride + 1
        # new_h = (data.shape[3]-self.kernal_size) // self.stride + 1 
        X_col = np.zeros((
            data.shape[0]*self.new_w*self.new_h, 
            data.shape[1]*self.kernal_size*self.kernal_size), 
        dtype=np.float32
        )

        im2col_c_func(data.tensor, X_col, data.shape[0], data.shape[1], data.shape[2], data.shape[3], self.kernal_size, self.stride)
        
        w_col = self.weight.tensor.reshape((self.out_channels, -1))
        
        out =  X_col @ w_col.T 
       
        self.conv_out.tensor = np.array(np.hsplit(out.T, data.shape[0])).reshape((data.shape[0], self.out_channels, self.new_h, self.new_w))
       
        self.cache = data, X_col, w_col

        def backward():
            X, X_col, w_col = self.cache
            m, _, _, _ = X.shape
            dout = self.conv_out.grad.reshape(self.conv_out.grad.shape[0] * self.conv_out.grad.shape[1], self.conv_out.grad.shape[2] * self.conv_out.grad.shape[3])
            dout = np.array(np.vsplit(dout, m))
            dout = np.concatenate(dout, axis=-1).T
            
            dX_col = dout @ w_col
            
            dw_col = (X_col.T @ dout).T
            
            data.grad = np.zeros(data.shape, dtype=np.float32)
            col2im_c_func(data.grad, dX_col, data.shape[0], data.shape[1], data.shape[2], data.shape[3], self.kernal_size, self.stride)
           
            self.weight.grad = dw_col.reshape((dw_col.shape[0], self.in_channels, self.kernal_size, self.kernal_size))

        self.conv_out._backward = backward
        return self.conv_out

class MaxPool2d:
        def __init__(self, size, stride):
            self.size = size
            self.stride = stride
        
        # @profile
        def __call__(self, data: Tensor) -> Tensor:
            backward_list.append(data)
            N, C, W, H = data.shape
            new_w = ((W-self.size) // self.stride) + 1
            new_h = ((H-self.size) // self.stride) + 1
            self.out = Tensor(np.zeros((N, C, new_w, new_h), dtype=np.float32), 'Maxpool output')
            self.max_idx = np.zeros((N, C, new_w, new_h), dtype=np.int32)
            maxpool_forward_c_func(
                data.tensor, 
                self.out.tensor, 
                self.max_idx, 
                data.shape[0], 
                data.shape[1], 
                data.shape[2], 
                data.shape[3], 
                self.size, 
                self.stride
            ) 
            
            def backward():
                data.grad = np.zeros(data.shape, dtype=np.float32)
                maxpool_backward_c_func(
                    data.tensor, 
                    data.grad, 
                    self.out.grad, 
                    self.max_idx, 
                    data.shape[0], 
                    data.shape[1], 
                    data.shape[2], 
                    data.shape[3], 
                    self.size, 
                    self.stride
                )
        
            self.out._backward = backward

            return self.out