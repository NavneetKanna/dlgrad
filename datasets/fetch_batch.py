from datasets import get_mnist
from dlgrad.tensor import Tensor


def fetch_batch(batch_size):
    i = 0
    x_train, y_train, _ , _ = get_mnist()
    x_batch_train = x_train[i:i+batch_size]
    Tensor(x_batch_train)
    y_batch_train = y_train[i:i+batch_size]
    Tensor(y_batch_train)
    i += batch_size
    return x_batch_train, y_batch_train
