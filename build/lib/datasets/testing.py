import idx2numpy
import dlgrad.tensor as tensor 

def get_mnist():
    x_train_arr = tensor(idx2numpy.convert_from_file(r'mnist/train-images-idx3-ubyte'))
    print(x_train_arr.shape())
    y_train_arr = tensor(idx2numpy.convert_from_file(r'datasets/mnist/train-labels-idx1-ubyte'))
    x_test_arr = tensor(idx2numpy.convert_from_file(r'datasets/mnist/t10k-images-idx3-ubyte'))
    y_test_arr = tensor(idx2numpy.convert_from_file(r'datasets/mnist/t10k-labels-idx1-ubyte'))

    # return x_train_arr, y_train_arr, x_test_arr, y_test_arr
get_mnist()