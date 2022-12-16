import idx2numpy
from dlgrad.tensor import Tensor 

def fetch_mnist():
    x_train = idx2numpy.convert_from_file(r'mnist/train-images-idx3-ubyte')
    x_train = Tensor(x_train.reshape(-1, 28*28))

    y_train = idx2numpy.convert_from_file(r'mnist/train-labels-idx1-ubyte')
    y_train = Tensor(y_train.reshape(-1, 28*28))

    x_test = idx2numpy.convert_from_file(r'mnist/t10k-images-idx3-ubyte')
    x_test = Tensor(x_test.reshape(-1, 28*28))

    y_test = idx2numpy.convert_from_file(r'mnist/t10k-labels-idx1-ubyte')
    y_test = Tensor(x_test.reshape(-1, 28*28))

    return x_train, y_train, x_test, y_test

def get_mnist_batch(BS):
    ...

