import numpy as np
from .tensor import Tensor
from .graph import CG
from .helper import backward_list
import numba as nb



def ReLU(matrix: Tensor) -> Tensor:
    backward_list.append(matrix)
    output = Tensor(np.maximum(0, matrix.tensor), 'Relu')

    # if not CG.stop_processing: CG.add_nodes('ReLU', output.tensor, matrix.tensor)

    def backward():
        
        matrix.tensor[matrix.tensor <= 0] = 0
        matrix.tensor[matrix.tensor > 0] = 1
        matrix.grad = (matrix.tensor * output.grad)


    output._backward = backward

    return output

def softmax(matrix: Tensor) -> np.ndarray:
    """
    We are subtracting each row with the maximum element, a kind of normalization,
    because the exp can get huge.
    """
    max_of_row = np.amax(matrix.tensor, axis=1, keepdims=True)
    matrix_exp = np.exp(matrix.tensor - max_of_row)
    matrix_sum = np.sum(matrix_exp, axis=1, keepdims=True)
    result = matrix_exp / matrix_sum

    return result