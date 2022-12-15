import numpy as np
from smallgrad.afu import softmax


def crossentropy(predictions, targets):
    """

    Similar to PyTorch cross entropy.
    First log softmax is done and then 
    negative log likehood is performed.
    Hence the targets must be logits, which means
    they must numbers(the values after performing the 
    weighted sum of the output layer) and not probabilities.
    And targets should "not" be one-hot encoded.
    """
    log_probs = np.log(softmax(predictions))
    nll = -(log_probs[range(len(targets)), targets].mean())
    return nll

