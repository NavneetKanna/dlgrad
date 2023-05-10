import numpy as np
from .afu import softmax
from .tensor import Tensor
from .graph import CG
from .helper import backward_list

def crossentropy(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Similar to PyTorch cross entropy.
    First log softmax is done and then 
    negative log likehood is performed.
    Hence the targets must be logits, which means
    they must numbers(the values after performing the 
    weighted sum of the output layer) and not probabilities.
    And targets should "not" be one-hot encoded.
    """
    backward_list.append(predictions)
    # Tensor.save_for_backward.append(predictions)

    one_hot_labels = np.zeros(predictions.shape)
    one_hot_labels[range(predictions.shape[0]), targets.tensor.T] = 1

    eps = 1e-10
    loss = -np.sum(one_hot_labels * np.log(softmax(predictions)+eps))
 
    # loss = -np.sum(one_hot_labels * np.log(softmax(predictions)))
    out = Tensor(loss/targets.shape[0])

    if not CG.stop_processing: CG.add_nodes('loss', out.tensor, predictions.tensor)

    # dL/dpreddictions = predictions-true(one-hot)
    def backward():
        # one_hot_labels = np.zeros(predictions.shape)
        # one_hot_labels[range(predictions.shape[0]), targets.tensor.T] = 1
        predictions.grad = (softmax(predictions) - one_hot_labels)

    out._backward = backward
        
    return out 



