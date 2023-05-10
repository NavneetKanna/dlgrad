import numpy as np
from .tensor import Tensor
from .graph import CG
from .helper import backward_list

# TODO: Tensor.maximum ?
def ReLU(matrix: Tensor):
    backward_list.append(matrix)
    output = Tensor(np.maximum(0, matrix.tensor))
    # Tensor.save_for_backward.append(matrix)

    if not CG.stop_processing: CG.add_nodes('ReLU', output.tensor, matrix.tensor)

    def backward():
        matrix.tensor[matrix.tensor <= 0] = 0
        matrix.tensor[matrix.tensor > 0] = 1
        matrix.grad = (matrix.tensor * output.grad)

    output._backward = backward

    return output

def softmax(matrix):
    """
    We are subtracting each row with the maximum element, a kind of normalization,
    because the exp can get huge.
    """
    try:
        max_of_row = np.amax(matrix.tensor, axis=1, keepdims=True)
        matrix_exp = np.exp(matrix.tensor-max_of_row)
        matrix_sum = np.sum(matrix_exp, axis=1, keepdims=True)
        result = matrix_exp / matrix_sum
    except RuntimeWarning:
        print("in softmax")
    return result