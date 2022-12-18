import sys
import os


# Wasted one full day on this !!!!

# os.chdir(r"C:\\Users\\navne\Documents\\vs_code\\dlgrad_main")

os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/")

sys.path.append(os.getcwd())
from dlgrad.tensor import Tensor
from dlgrad.nn import MLP
from datasets.fetch_mnist import fetch_mnist
from dlgrad.afu import ReLU, softmax 
from nnn.training import train
from dlgrad.graph import draw_cg

class Net:
    def __init__(self) -> None:
        # self.x_train, self.y_train, self.x_test, self.y_test = fetch_mnist()
        self.batch_size = 32
        self.epochs = 10

        self.fc1 = MLP(28*28, 64)
        self.fc2 = MLP(64, 10)

    # TODO: get rid of this Tensor()
    def forward(self, x_train):
        x = self.fc1(Tensor(x_train))
        x = ReLU(x)
        x = self.fc2(x)
        # x = softmax(x)

        return x


net = Net()
x_train, y_train, x_test, y_test = fetch_mnist()
net.forward(x_train.tensor[0:32])
draw_cg()

# def main():
#     epochs = 10
#     x_train, y_train, x_test, y_test = fetch_mnist()
#     BS = 32

#     for _ in range(epochs):
#         idx = 0
#         for _ in range(x_train.shape()[0]/BS):
#             x_train = x_train[idx:idx+BS]
#             y_train = y_train[idx:idx+BS]
#             idx += BS
#             train(net, x_train, y_train)


# if __name__ == '__main__':
#     main()


# from cmath import tanh
# import tensorflow as tf
# import numpy as np

# y_true = [3, 3, 1]
# y_pred = [
#     [0.3377, 0.4867, 0.8842, 0.0854, 0.2147],
#     [0.4853, 0.0468, 0.6769, 0.5482, 0.1570],
#     [0.0976, 0.9899, 0.6903, 0.0828, 0.0647]
# ]
# scce3 = tf.keras.losses.SparseCategoricalCrossentropy()
# loss3 = scce3(y_true, tf.nn.softmax(y_pred)).numpy()
# print(f"tf {loss3}")


# from torch import nn
# import torch
# loss = nn.CrossEntropyLoss()
# y_true = torch.Tensor([3, 3, 1]).long()
# y_pred = torch.Tensor([
#     [0.3377, 0.4867, 0.8842, 0.0854, 0.2147],
#     [0.4853, 0.0468, 0.6769, 0.5482, 0.1570],
#     [0.0976, 0.9899, 0.6903, 0.0828, 0.0647]
# ])
# loss2 = loss(y_pred, y_true)
# print(f"torch {loss2}")


# import numpy as np
# from smallgrad.afu import softmax

# def cross_entropy(predictions, targets):
#     log_probs = np.log(softmax(predictions))

#     nll = -(log_probs[range(len(targets)), targets].mean())
#     return nll

#     # N = predictions.shape[0]
#     # ce = -np.sum(targets * np.log(predictions))/N 
#     # return ce

# predictions = np.asarray([
#     [0.3377, 0.4867, 0.8842, 0.0854, 0.2147],
#     [0.4853, 0.0468, 0.6769, 0.5482, 0.1570],
#     [0.0976, 0.9899, 0.6903, 0.0828, 0.0647]
# ])
# # targets = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])
# targets = np.array([3, 3, 1])



# print(f"mine loss {cross_entropy(predictions, targets)}")
# print(f"mine {np.sum(cross_entropy(softmax(predictions), targets))/3}")