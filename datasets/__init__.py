import idx2numpy

def get_mnist():
    print("get mnist")
    x_train_arr = idx2numpy.convert_from_file(r'datasets/mnist/train-images-idx3-ubyte')
    y_train_arr = idx2numpy.convert_from_file(r'datasets/mnist/train-labels-idx1-ubyte')
    x_test_arr = idx2numpy.convert_from_file(r'datasets/mnist/t10k-images-idx3-ubyte')
    y_test_arr = idx2numpy.convert_from_file(r'datasets/mnist/t10k-labels-idx1-ubyte')

    return x_train_arr, y_train_arr, x_test_arr, y_test_arr
