import idx2numpy
from dlgrad.tensor import Tensor 

def fetch_mnist():
    x_train = idx2numpy.convert_from_file(r'datasets/mnist/train-images-idx3-ubyte')
    x_train = Tensor(x_train.reshape((-1, 28*28)))

    y_train = idx2numpy.convert_from_file(r'datasets/mnist/train-labels-idx1-ubyte')
    y_train = Tensor(y_train)
    # print(f"y_train shape {y_train.shape}")

    x_test = idx2numpy.convert_from_file(r'datasets/mnist/t10k-images-idx3-ubyte')
    x_test = Tensor(x_test.reshape((-1, 28*28)))
    # print(f"x_test shape {x_test.shape}")

    y_test = idx2numpy.convert_from_file(r'datasets/mnist/t10k-labels-idx1-ubyte')
    y_test = Tensor(y_test)
    # print(f"y_test shape {y_test.shape}")

    return x_train, y_train, x_test, y_test

def get_mnist_batch(BS):
    ...

