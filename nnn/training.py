from dlgrad.loss import crossentropy
from dlgrad.tensor import Tensor
from dlgrad.graph import display_graph
from dlgrad.afu import softmax
import numpy as np
from datasets.fetch_mnist import fetch_mnist
from sklearn.metrics import accuracy_score

lo_val = 0
ff = True
a = []
loss_graph = []
acc_graph = []


def train(model, x_train, y_train, flag, epoch, lr=1e-3):
    global lo_val, ff, acc_graph, loss_graph

    x = model.forward(x_train, flag)
    pred = x
   
    loss = crossentropy(x, y_train, flag)
   
    if ff: 
        display_graph()
        ff = False
    lo_val += 1

    loss.backward()

    for parameters in Tensor.get_parameters():
        parameters.tensor = parameters.tensor - (lr*parameters.grad)

    if lo_val % 100 == 0:
        acc = accuracy_score(y_train.tensor.T, np.argmax(softmax(pred), axis=1))
        acc_graph.append(acc*100)
        loss_graph.append(loss.tensor)

    # Calculate accuracy and loss 
    if lo_val == 1874:
        acc = accuracy_score(y_train.tensor.T, np.argmax(softmax(pred), axis=1))
        print("true")
        print(y_train.tensor.T)
        print("pred")
        print(np.argmax(softmax(pred), axis=1))
        print(f"acc {acc*100}%")
        print(f"loss {loss.tensor}")
        lo_val = 0
