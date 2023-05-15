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

        # (16, 3, 4, 4)
        self.kernel= np.random.random((self.out_channels, self.in_channels, self.kernal_size, self.kernal_size))
        # (16, 1, 48)
        self.im2col_kernel = np.random.randn(self.out_channels, 1, (self.kernal_size**2)*self.in_channels)

    def __call__(self, data): 
        self.new_w = ((data.shape[-2] - self.kernal_size + 2*0)//self.stride) + 1 
        self.new_h = ((data.shape[-1] - self.kernal_size + 2*0)//self.stride) + 1 
        self.conv_out = np.zeros((data.shape[0], self.out_channels, self.new_w, self.new_h), dtype=np.float32)
        return self.conv(data)

    def im2col(self, data):
        res = np.zeros((self.kernal_size*self.kernal_size*data.shape[0], self.new_w*self.new_h))

        idx = 0
        for row in range(0, data.shape[-2]-self.kernal_size+ 1, self.stride):
            for col in range(0, data.shape[-1]-self.kernal_size + 1, self.stride):
                patch = data[..., row:row+self.kernal_size, col:col+self.kernal_size].reshape(-1)
                res[:, idx] = patch
                idx += 1
        return res.T

    def conv(self, data):
        for bs in range(data.shape[0]):
            for no_of_filter in range(self.out_channels):
                res = self.im2col(data[bs, ...])
                res = res @ self.im2col_kernel[no_of_filter].T
                res = res.reshape(self.new_w, self.new_h).round(4) # col2img
                self.conv_out[bs, no_of_filter, ...] = res
            
        return self.conv_out 


class MaxPool2D:
    def __init__(self) -> None:
        pass