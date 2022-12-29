import numpy as np
from .afu import softmax
from .tensor import Tensor
from .graph import draw_graph
import torch
from torch import nn
import warnings

def crossentropy(predictions: Tensor, targets: Tensor, flag=False):
    """
    Similar to PyTorch cross entropy.
    First log softmax is done and then 
    negative log likehood is performed.
    Hence the targets must be logits, which means
    they must numbers(the values after performing the 
    weighted sum of the output layer) and not probabilities.
    And targets should "not" be one-hot encoded.
    """
    Tensor.save_for_backward.append(predictions)

    one_hot_labels = np.zeros(predictions.shape)
    one_hot_labels[range(predictions.shape[0]), targets.tensor.T] = 1

    eps = 1e-10
    loss = -np.sum(one_hot_labels * np.log(softmax(predictions)+eps))
 
    # loss = -np.sum(one_hot_labels * np.log(softmax(predictions)))
    out = Tensor(loss/targets.shape[0])

    if flag:
        draw_graph(
            'Cross-Entropy Loss',
            (out.tensor.shape, 'Loss'),
            (predictions.shape, 'Predictions')
        )

    # dL/dpreddictions = predictions-true(one-hot)
    def backward():
        # one_hot_labels = np.zeros(predictions.shape)
        # one_hot_labels[range(predictions.shape[0]), targets.tensor.T] = 1
        predictions.grad = (softmax(predictions) - one_hot_labels)

    out._backward = backward
        
    return out 



