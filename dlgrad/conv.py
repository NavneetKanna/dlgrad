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

    def conv(self, data):
        size = ((data.shape[2] - self.kernal_size + 2*self.padding) / self.stride) + 1
        self.output = np.zeros((data.shape[0], self.filters.shape[0], int(size), int(size)))
        # for each 4 batch size
        for batch in range(data.shape[0]):
            # for each 6 filters
            for total_no_of_filters in range(self.out_channels):
                out_row = 0
                for row in range(0, (data.shape[2]-self.kernal_size+1), self.stride):
                    out_col = 0
                    for col in range(0, (data.shape[2]-self.kernal_size+1), self.stride):
                        sum = 0
                        # for either rgb or grayscale
                        for filter in range(self.in_channels):
                            sum += np.sum(data[batch, filter,  row:row+self.kernal_size, col:col+self.kernal_size] * self.filters).round(4)
                        self.output[batch][total_no_of_filters][out_row][out_col] = np.round(sum, 4) 
                        out_col += 1
                    out_row += 1
        return self.output 