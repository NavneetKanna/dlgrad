import sys
import os
# os.chdir(r"C:\\Users\\navne\Documents\\vs_code\\dlgrad_main")
os.chdir(r"/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/")
sys.path.append(os.getcwd())
from dlgrad.tensor import Tensor
from dlgrad.nn import MLP
from datasets.fetch_mnist import fetch_mnist
from dlgrad.afu import ReLU, softmax 
from nnn.training import train
from dlgrad.graph import display_graph 


class Net:
    def __init__(self) -> None:
        # self.x_train, self.y_train, self.x_test, self.y_test = fetch_mnist()
        self.batch_size = 32
        self.epochs = 10

        self.fc1 = MLP(28*28, 64, bias=True)
        self.fc2 = MLP(64, 10, bias=True)

    def forward(self, x_train):
        x = self.fc1(x_train)
        x = ReLU(x)
        x = self.fc2(x)
        # x = softmax(x)
        return x

# net = Net()
# x_train, y_train, x_test, y_test = fetch_mnist()
# print(y_train)
# net.forward(x_train[0:32])
# display_graph()

def main():
    epochs = 1
    x_train, y_train, x_test, y_test = fetch_mnist()
    # print(f"x train from first {x_train.shape}")
    BS = 32

    net = Net()

    for e in range(epochs):
        idx = 0
        for u in range(int(x_train.shape[0]/BS)):
            X_train = x_train[idx:idx+BS]
            Y_train = y_train[idx:idx+BS]
            idx += BS
            if (u % 500) == 0:
                train(net, X_train, Y_train, pp=True)
            else:
                train(net, X_train, Y_train)

        print(f"epoch {e}")

if __name__ == '__main__':
    main()


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



# one_hot_labels = torch.zeros(3, 5) # 3 examples, 3 classes
# one_hot_labels[range(3), y_true] = 1
# scce3 = tf.keras.losses.SparseCategoricalCrossentropy()
# loss3 = scce3(y_true, tf.nn.softmax(y_pred)).numpy()
# print(f"tf {loss3}")


# from torch import nn
# import torch
# loss = nn.CrossEntropyLoss()
# # input = torch.randn(3, 5, requires_grad=True)
# # print("input")
# # print(input)
# # target = torch.empty(3, dtype=torch.long).random_(5)
# # print("target")
# # print(target)


# # y_true = torch.Tensor([3, 3, 1]).long()
# # y_pred = torch.Tensor([
# #     [0.3377, 0.4867, 0.8842, 0.0854, 0.2147],
# #     [0.4853, 0.0468, 0.6769, 0.5482, 0.1570],
# #     [0.0976, 0.9899, 0.6903, 0.0828, 0.0647]
# # ])

# y_true = torch.tensor([3, 4, 4]).long()
# y_pred = torch.tensor([[-1.3211,  0.9844, -0.9693, -0.1271, -1.1033],
#         [-0.2364, -0.3588, -0.0036,  0.3900,  1.0710],
#         [-0.6053,  0.3890,  0.6630,  1.5039,  0.1959]])

# loss2 = loss(y_pred, y_true)
# print(f"torch {loss2}")



# one_hot_labels = torch.zeros(3, 5) # 3 examples, 3 classes
# one_hot_labels[range(3), y_true] = 1
# loss2 = loss(y_pred, one_hot_labels)
# print(f"torch one hot loss {loss2}")



# import numpy as np
# from dlgrad.afu import softmax

# def cross_entropy(predictions, targets):
#     log_probs = np.log(softmax(predictions))
#     nll = -(log_probs[range(targets.shape[0]), targets].mean(axis=0))
#     return nll

#     # N = predictions.shape[0]
#     # ce = -np.sum(targets * np.log(predictions))/N 
#     # return ce


# predictions = np.asarray([[-1.3211,  0.9844, -0.9693, -0.1271, -1.1033],
#         [-0.2364, -0.3588, -0.0036,  0.3900,  1.0710],
#         [-0.6053,  0.3890,  0.6630,  1.5039,  0.1959]])
# targets = np.array([3, 4, 4])


# # predictions = np.asarray([
# #     [0.3377, 0.4867, 0.8842, 0.0854, 0.2147],
# #     [0.4853, 0.0468, 0.6769, 0.5482, 0.1570],
# #     [0.0976, 0.9899, 0.6903, 0.0828, 0.0647]
# # ])
# # # targets = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])
# # targets = np.array([3, 3, 1])


# log_probs = np.log(softmax(predictions))
# print("log probs")
# print(log_probs)
# print("range")
# a = log_probs[range(targets.shape[0]), targets)]
# print(a)
# print("mean")
# print(log_probs[range(targets.shape[0]), targets].mean(axis=0))


# print(f"mine loss {cross_entropy(predictions, targets)}")
# print(f"mine one hot loss {cross_entropy(predictions, one_hot_labels)}")
# print(f"mine {np.sum(cross_entropy(softmax(predictions), targets))/3}")