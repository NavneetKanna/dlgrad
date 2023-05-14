import numpy as np


class Conv2D:
    # nn.conv2d(in_channels, out_channels, kernel_size) = (out_channels, in_channels, kernel_size, kernel_size)
    # nn.conv2d(3, 6, 5) = (6, 3, 5, 5)
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        self.in_channels = in_channels # No of Channles
        self.out_channels = out_channels # No of Filters/Kernels
        self.kernal_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.filters = np.random.random((self.out_channels, self.in_channels, self.kernal_size, self.kernal_size))

    def __call__(self, data): 
        self.conv(data)

    def im2col(self, x):
        self.new_w = ((x.shape[2] - self.kernal_size + 2*0)//self.stride) + 1 
        self.new_h = ((x.shape[1] - self.kernal_size + 2*0)//self.stride) + 1 

        res = np.zeros((self.kernal_size*self.kernal_size*x.shape[0], self.new_w*self.new_h))

        idx = 0
        for row in range(0, x.shape[-2]-self.kernal_size+ 1, self.stride):
            for col in range(0, x.shape[-1]-self.kernal_size + 1, self.stride):
                patch = x[..., row:row+self.kernal_size, col:col+self.kernal_size].reshape(-1)
                res[:, idx] = patch
                idx += 1
        return res.T

    def conv(self, data):
        res = self.im2col(data[0, ...])
        final = res @ self.filters.T
        fi = final.reshape(self.new_w, self.new_h).round(4)
        # for channel in self.in_channels:
        #     res = self.im2col(data[channel, ...])
        #     final = res @ self.filters.T
        #     fi = final.reshape(self.new_w, self.new_h).round(4)

        return fi