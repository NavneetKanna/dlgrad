import numpy as np
from .afu import softmax
from .tensor import Tensor


def crossentropy(predictions: Tensor, targets: Tensor):
    """
    Similar to PyTorch cross entropy.
    First log softmax is done and then 
    negative log likehood is performed.
    Hence the targets must be logits, which means
    they must numbers(the values after performing the 
    weighted sum of the output layer) and not probabilities.
    And targets should "not" be one-hot encoded.
    """
    log_probs = np.log(softmax(predictions.tensor))
    nll = -(log_probs[range(targets.shape[0]), targets.tensor.T])
    out = Tensor(nll)

    # dL/dpreddictions = predictions-true(one-hot)
    def backward():
        one_hot_labels = Tensor.zeros(predictions.shape[0], predictions.shape[1]) 
        one_hot_labels[range(predictions.shape[0]), targets] = 1
        predictions.grad += predictions - one_hot_labels
    
    out.backward = backward
        
    return out 



