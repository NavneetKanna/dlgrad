import numpy as np
from .tensor import Tensor
from .graph import draw_graph 


def ReLU(matrix, flag):
    
    output = Tensor(np.maximum(0, matrix.tensor))
    Tensor.save_for_backward.append(matrix)

    if flag:
        draw_graph(
            'Relu',
            (output.tensor.shape, 'output', output.grad.shape),
            (matrix.tensor.shape, 'input', matrix.grad.shape)
        )

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
    max_of_row = np.amax(matrix.tensor, axis=1, keepdims=True)
    matrix_exp = np.exp(matrix.tensor-max_of_row)
    matrix_sum = np.sum(matrix_exp, axis=1, keepdims=True)
    result = matrix_exp / matrix_sum
    return result