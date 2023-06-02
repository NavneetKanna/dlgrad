import numpy as np
from .tensor import Tensor
from .helper import backward_list
from numpy.lib.stride_tricks import as_strided
import numba as nb

# https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays


@nb.njit
def _im2col(data, kernal_size, stride):
    new_w = (data.shape[2]-kernal_size) // stride + 1
    new_h = (data.shape[3]-kernal_size) // stride + 1 
    res = np.zeros((data.shape[0]*new_w*new_h, data.shape[1]*kernal_size*kernal_size))
    idx = 0

    for bs in range(data.shape[0]):
        for row in range(0, data.shape[-2]-kernal_size+ 1, stride):
            for col in range(0, data.shape[-1]-kernal_size + 1, stride):
                patch1 = data[bs, ..., row:row+kernal_size, col:col+kernal_size]
                patch2 = patch1.reshape(-1)
                res[idx, :] = patch2.T
                idx += 1
                
    return res

@nb.njit
def _col2im(data, dx, c, kernal_size, stride):
    idx = 0
    for bs in range(data.shape[0]):
        for row in range(0, data.shape[-2]-kernal_size+ 1, stride):
            for col in range(0, data.shape[-1]-kernal_size + 1, stride):
                data[bs, ..., row:row+kernal_size, col:col+kernal_size] += dx[idx, : ].reshape(c, row+kernal_size-row, col+kernal_size - col)
                idx += 1 

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

        self.weight = Tensor(np.random.random((self.out_channels, self.in_channels, self.kernal_size, self.kernal_size)).astype(np.float32))
        
    def __call__(self, data: Tensor) -> Tensor: 
        # print(f"conv data {data.tensor.shape}")
        backward_list.extend((data, self.weight))
        self.new_w = ((data.shape[-2] - self.kernal_size + 2*self.padding)//self.stride) + 1 
        self.new_h = ((data.shape[-1] - self.kernal_size + 2*self.padding)//self.stride) + 1 
        self.conv_out = Tensor(np.zeros((data.shape[0], self.out_channels, self.new_w, self.new_h), dtype=np.float32))
        data.grad = np.zeros(data.tensor.shape, dtype=np.float32)
        return self.conv(data)

    @profile 
    def conv(self, data) -> Tensor:

        X_col = self.im2col(data, self.kernal_size, self.stride)
        w_col = self.weight.tensor.reshape((self.out_channels, -1))
        # print(f"x_col {X_col.shape} w_col {w_col.shape}")
        # Perform matrix multiplication.
        out = w_col @ X_col.T 
        # print(f"out {out.shape}")

        self.conv_out.tensor = np.array(np.hsplit(out, data.shape[0])).reshape((data.shape[0], self.out_channels, self.new_h, self.new_w))
        # print(f"conv_out.tensor {self.conv_out.tensor.shape}")
        self.cache = data, X_col, w_col

        @profile
        def backward():
            X, X_col, w_col = self.cache
            m, _, _, _ = X.shape
            dout = self.conv_out.grad.reshape(self.conv_out.grad.shape[0] * self.conv_out.grad.shape[1], self.conv_out.grad.shape[2] * self.conv_out.shape[3])
            dout = np.array(np.vsplit(dout, m))
            dout = np.concatenate(dout, axis=-1)
            dX_col = w_col.T @ dout
            dw_col = dout @ X_col.T
            self.col2im(data.grad, dX_col, self.out_channels, self.kernal_size, self.stride)
            self.weight.grad = dw_col.reshape((dw_col.shape[0], self.out_channels, self.kernal_size, self.kernal_size))

        self.conv_out.backward = backward

        return self.conv_out

    # @profile 
    # TODO: Pad cols if stride is more than 1 
    # IMP: ALL ROWS CONTAIN THE data
    def im2col(self, data, kernal_size, stride):
        new_w = (data.shape[2]-kernal_size) // stride + 1
        new_h = (data.shape[3]-kernal_size) // stride + 1 
        res = np.zeros((data.shape[0]*new_w*new_h, data.shape[1]*kernal_size*kernal_size))
        idx = 0

        for bs in range(data.shape[0]):
            for row in range(0, data.shape[-2]-kernal_size+ 1, stride):
                for col in range(0, data.shape[-1]-kernal_size + 1, stride):
                    patch1 = data[bs, ..., row:row+kernal_size, col:col+kernal_size]
                    patch2 = patch1.reshape(-1)
                    res[idx, :] = patch2.T
                    idx += 1
                    
        return res

    # @profile 
    # def col2img(self, matrix, w, h):
    def col2im(self, data, dx, c, kernal_size, stride):
        idx = 0
        for bs in range(data.shape[0]):
            for row in range(0, data.shape[-2]-kernal_size+ 1, stride):
                for col in range(0, data.shape[-1]-kernal_size + 1, stride):
                    data[bs, ..., row:row+kernal_size, col:col+kernal_size] += dx[idx, : ].reshape(c, row+kernal_size-row, col+kernal_size - col)
                    idx += 1 

'''
N, C, W, H = data.shape
    new_w = ((W-size) // stride) + 1
    new_h = ((H-size) // stride) + 1
    out1 = np.zeros((N, C, new_w, new_h))
    
    for bs in range(N):
        for channel in range(C):
            # print("--------------")
            da = data[bs, channel]
            t = as_strided(da, shape=(new_w, new_h, size, size), strides=(data.strides[-2]*stride, data.strides[-1]*stride, data.strides[-2], data.strides[-1]))
            # print(t)
            # print(t.shape)
            b = np.amax(t, axis=(-1, -2)).round(3)
            # print(b)
            out1[bs, channel] = b


'''
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
            self.out = Tensor(np.zeros((N, C, new_w, new_h), dtype=np.float32))

            for bs in range(N):
                for channel in range(C):
                    da = data[bs, channel]
                    t = as_strided(da, shape=(new_w, new_h, self.size, self.size), 
                                   strides=(data.tensor.strides[-2]*self.stride, data.tensor.strides[-1]*self.stride, data.tensor.strides[-2], data.tensor.strides[-1])
                        )
                    b = np.amax(t, axis=(-1, -2)).round(3)
                    self.out.tensor[bs, channel] = b

            # for bs in range(N):
            #     for channel in range(C):
            #         for row in range(0, H, self.stride):
            #             if H-row < self.size:
            #                 continue
            #             for col in range(0, W, self.stride):
            #                 if W-col < self.size:
            #                     continue
            #                 pool_slice = data[bs, channel, row:row+self.size, col:col+self.size]
            #                 self.out.tensor[bs, channel, row//self.stride, col//self.stride] = np.max(pool_slice)
            #                 max_index = np.argmax(pool_slice)
            #                 max_index = np.unravel_index(max_index, pool_slice.shape)
            #                 self.switch_row[bs, channel, row // self.stride, col // self.stride] = max_index[0] + row
            #                 self.switch_col[bs, channel, row // self.stride, col // self.stride] = max_index[1] + col
            # @profile
            def backward():
                # N, C, new_w, new_h = self.out.grad.shape
                data.grad = np.zeros(data.shape, dtype=np.float32)

                for bs in range(data.grad.shape[0]):
                    for channel in range(data.grad.shape[1]):
                        ridx = 0
                        for row in range(0, data.grad.shape[-2], self.stride):
                            cidx = 0
                            if data.grad.shape[-2]-row < self.size:
                                continue
                            for col in range(0, data.grad.shape[-1], self.stride):
                                if data.grad.shape[-1]-col < self.size:
                                    continue
                                t = data[bs, channel, row:row+self.size, col:col+self.size]
                                g = data.grad[bs, channel, row:row+self.size, col:col+self.size]
                                h = self.out.grad[bs, channel, ridx, cidx]
                                y = np.unravel_index(np.argmax(t), t.shape)
                                np.add.at(g, y, h)
                                cidx += 1
                            ridx += 1
        
                # for bs in range(N):
                #     for channel in range(C):
                #         for row in range(new_h):
                #             for col in range(new_w):
                #                 pool_row = self.switch_row[bs, channel, row, col]
                #                 pool_col = self.switch_col[bs, channel, row, col]
                #                 data.grad[bs, channel, pool_row, pool_col] = self.out.grad[bs, channel, row, col]

            self.out._backward = backward

            return self.out