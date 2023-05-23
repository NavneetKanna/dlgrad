import numpy as np
from .tensor import Tensor
from .helper import backward_list

# https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays

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

        # (16, 6, 5, 5)
        self.weight = Tensor(np.random.random((self.out_channels, self.in_channels, self.kernal_size, self.kernal_size)).astype(np.float32))
        
    def __call__(self, data: Tensor) -> Tensor: 
        backward_list.extend((data, self.weight))
        self.new_w = ((data.shape[-2] - self.kernal_size + 2*0)//self.stride) + 1 
        self.new_h = ((data.shape[-1] - self.kernal_size + 2*0)//self.stride) + 1 
        # (4, 16, 10, 10)
        self.conv_out = Tensor(np.zeros((data.shape[0], self.out_channels, self.new_w, self.new_h), dtype=np.float32))
        data.grad = np.zeros(data.tensor.shape)
        return self.conv(data)

    def conv(self, data) -> Tensor:
        self.kernel = Tensor(self.weight.tensor)
        self.kernel.tensor = self.kernel.tensor.reshape(-1, (self.kernal_size**2)*self.in_channels)

        for bs in range(data.shape[0]):
            res = self.im2col(data[bs, ...])
            # Dont transpose bcs in matmul it is being done
            res_out = Tensor.matmul(res, self.kernel)
            self.matmul_output.append((res_out, res, self.kernel))
            res_out = self.col2img(res_out, self.new_w, self.new_h)
            self.conv_out.tensor[bs, ...] = res_out
        
        def backward():
            BS = self.conv_out.shape[0]
            # (4, 16, 10, 10)
            self.conv_out.grad = self.conv_out.grad.reshape(BS, self.out_channels, self.new_w, self.new_h)
            # assert len(matmul_output) == BS
            for bs, idx in zip(range(BS), range(len(self.matmul_output))):
                t = self.conv_out.grad[bs, ...]
                # (100, 16)
                t = t.reshape(t.shape[0], -1).T

                self.matmul_output[idx][0]._backward('conv', t)

                # Update grad of weight/filter/kernel
                # (16, 6, 5, 5)
                self.weight.grad += self.matmul_output[idx][2].grad.reshape(self.weight.shape)

                # Update grad of input
                # (6, 14, 14)
                t = self.col2im_back(self.matmul_output[idx][1].grad, self.new_w, self.new_h, self.stride, self.kernal_size, self.kernal_size, data.shape[1])
                
                data.grad[bs, ...] = t
            
            self.matmul_output.clear()

        self.conv_out._backward = backward

        return self.conv_out

    def im2col(self, data) -> Tensor:
        res = np.zeros((self.kernal_size*self.kernal_size*data.shape[0], self.new_w*self.new_h))
        idx = 0
        for row in range(0, data.shape[-2]-self.kernal_size+ 1, self.stride):
            for col in range(0, data.shape[-1]-self.kernal_size + 1, self.stride):
                patch = data[..., row:row+self.kernal_size, col:col+self.kernal_size].reshape(-1)
                res[:, idx] = patch
                idx += 1
        return Tensor(res.T)

    def col2img(self, matrix, w, h):
        out = np.zeros((matrix.shape[1], w, h))
        for filter in range(matrix.shape[1]):
            col = matrix[:, filter]
            out[filter, ...] = col.reshape(w, h)
        return out

    def col2im_back(self, dim_col, h_prime, w_prime, stride, hh, ww, c):
        # 14
        H = (h_prime - 1) * stride + hh
        # 14
        W = (w_prime - 1) * stride + ww
        # (6, 14, 14)
        dx = np.zeros([c,H,W], dtype=np.float32)

        for i in range(h_prime*w_prime):
            # (150,)
            row = dim_col[i,:]
            h_start = int((i / w_prime) * stride)
            w_start = int((i % w_prime) * stride)
            # (6, 5, 5)
            dx[:, h_start:h_start+hh, w_start:w_start+ww] += np.reshape(row,(c,hh,ww))
        return dx

class MaxPool2d:
        def __init__(self, size, stride):
            self.size = size
            self.stride = stride

        def __call__(self, data) -> Tensor:
            backward_list.append(data)
            N, C, W, H = data.shape
            new_w = ((W-self.size) // self.stride) + 1
            new_h = ((H-self.size) // self.stride) + 1
            self.out = Tensor(np.zeros((N, C, new_w, new_h)))
            self.switch_row = np.zeros((N, C, new_w, new_h), dtype=int)
            self.switch_col = np.zeros((N, C, new_w, new_h), dtype=int)

            for bs in range(N):
                for channel in range(C):
                    for row in range(0, H, self.stride):
                        if H-row < self.size:
                            continue
                        for col in range(0, W, self.stride):
                            if W-col < self.size:
                                continue
                            pool_slice = data[bs, channel, row:row+self.size, col:col+self.size]
                            self.out.tensor[bs, channel, row//self.stride, col//self.stride] = np.max(pool_slice)
                            max_index = np.argmax(pool_slice)
                            max_index = np.unravel_index(max_index, pool_slice.shape)
                            self.switch_row[bs, channel, row // self.stride, col // self.stride] = max_index[0] + row
                            self.switch_col[bs, channel, row // self.stride, col // self.stride] = max_index[1] + col

            def backward():
                N, C, new_w, new_h = self.out.grad.shape
                data.grad = np.zeros(data.shape)
        
                for bs in range(N):
                    for channel in range(C):
                        for row in range(new_h):
                            for col in range(new_w):
                                pool_row = self.switch_row[bs, channel, row, col]
                                pool_col = self.switch_col[bs, channel, row, col]
                                data.grad[bs, channel, pool_row, pool_col] = self.out.grad[bs, channel, row, col]

            self.out._backward = backward

            return self.out