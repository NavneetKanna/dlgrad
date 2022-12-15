import numpy as np



def ReLU(matrix):
    return np.maximum(0, matrix)

def softmax(matrix):
    """
    We are subtracting each row with the maximum element, a kind of normalization,
    because the exp can get huge.
    """
    max_of_row = np.amax(matrix, axis=1, keepdims=True)
    matrix_exp = np.exp(matrix-max_of_row)
    matrix_sum = np.sum(matrix_exp, axis=1, keepdims=True)
    result = matrix_exp / matrix_sum
    return result