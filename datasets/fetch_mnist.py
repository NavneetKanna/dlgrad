import idx2numpy
import numpy as np
from dlgrad.tensor import Tensor 

def fetch_mnist():
    x_train = idx2numpy.convert_from_file(r'datasets/mnist/train-images-idx3-ubyte')
    x_train = x_train/255.0
    x_train = Tensor(x_train.reshape((-1, 28*28)))

    y_train = idx2numpy.convert_from_file(r'datasets/mnist/train-labels-idx1-ubyte')
    y_train = Tensor(y_train)

    x_test = idx2numpy.convert_from_file(r'datasets/mnist/t10k-images-idx3-ubyte')
    x_test = x_test/255.0
    x_test = Tensor(x_test.reshape((-1, 28*28)))

    y_test = idx2numpy.convert_from_file(r'datasets/mnist/t10k-labels-idx1-ubyte')
    y_test = Tensor(y_test)

    return x_train, y_train, x_test, y_test


