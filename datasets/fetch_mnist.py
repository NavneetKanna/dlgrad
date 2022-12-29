import idx2numpy
import numpy as np
from dlgrad.tensor import Tensor 

class MNIST:
    def __init__(self):
        self.idx = 0

        self.x_train = idx2numpy.convert_from_file(r'datasets/mnist/train-images-idx3-ubyte')
        self.x_train = self.x_train/255.0
        self.x_train = self.x_train.reshape((-1, 28*28))

        self.y_train = idx2numpy.convert_from_file(r'datasets/mnist/train-labels-idx1-ubyte')
        self.y_train = self.y_train

        self.x_test = idx2numpy.convert_from_file(r'datasets/mnist/t10k-images-idx3-ubyte')
        self.x_test = self.x_test/255.0
        self.x_test = self.x_test.reshape((-1, 28*28))

        self.y_test = idx2numpy.convert_from_file(r'datasets/mnist/t10k-labels-idx1-ubyte')
        self.y_test = self.y_test

    def get_train_data(self) -> np.ndarray:
        return self.x_train, self.y_train
    
    def get_test_data(self) -> Tensor:
        return Tensor(self.x_test), Tensor(self.y_test)

    def get_batch_data(self, x_data: np.ndarray, y_data: np.ndarray, BS: int) -> Tensor:
        x = Tensor(x_data[self.idx:self.idx+BS])
        y = Tensor(y_data[self.idx:self.idx+BS])
        self.idx += BS
        return x, y


