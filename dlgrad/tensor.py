from __future__ import annotations
import numpy as np
from .graph import CG
from .helper import backward_list


class Tensor:
    cg = CG()

    def __init__(self, data: np.ndarray, name=None):
        self.tensor: np.ndarray = data 
        self.shape = self.tensor.shape
        self._backward = lambda: None
        self.name = name
        self.grad = 0 

    @classmethod
    def uniform(cls, shape: tuple, start=-1, end=1):
        # TODO: Seems to be a bad way to do this as said by geohotz
        return cls(np.random.default_rng().uniform(start, end, shape).astype(np.float32))

    # Only used for - flatten all dimensions except batch
    @classmethod
    def flatten(cls, data: Tensor) -> Tensor:
        backward_list.append(data)
        out = Tensor(data.tensor.reshape(data.shape[0], -1))
        out.tensor.astype(np.float32)

        def backward():
            data.grad = out.grad.reshape(data.tensor.shape)
            data.grad.astype(np.float32)
        
        out._backward = backward
        
        return out
    
    def add(self: Tensor, other: Tensor) -> Tensor:
        backward_list.extend((self, other))
        output = Tensor(self.tensor + other.tensor, 'Add')

        # if not CG.stop_processing: CG.add_nodes('add', output.tensor, self.tensor, other.tensor)

        def backward():
            self.grad = output.grad
            other.grad = np.sum(output.grad, axis=0, keepdims=True, dtype=np.float32)
        
        output._backward = backward

        return output
    
    def matmul(self: Tensor, other: Tensor) -> Tensor:
        backward_list.extend((self, other))
        output = Tensor(self.tensor @ other.tensor.T, 'Matmul')

        # if not CG.stop_processing: CG.add_nodes('matmul', output.tensor, self.tensor, other.tensor)

        def backward():
            self.grad = output.grad @ other.tensor
            other.grad = (self.tensor.T @ output.grad).T 
            # TODO: WHY IS IT ALL 0 ? 

        output._backward = backward

        return output 

    def ReLU(self, matrix: Tensor) -> Tensor:
        backward_list.append(matrix)
        output = Tensor(np.maximum(0, matrix.tensor), 'Relu')

        # if not CG.stop_processing: CG.add_nodes('ReLU', output.tensor, matrix.tensor)

        def backward():
            matrix.tensor[matrix.tensor <= 0] = 0
            matrix.tensor[matrix.tensor > 0] = 1
            matrix.grad = (matrix.tensor * output.grad)

        output._backward = backward

        return output

    def softmax(self, matrix: Tensor) -> np.ndarray:
        """
        We are subtracting each row with the maximum element, a kind of normalization,
        because the exp can get huge.
        """
        max_of_row = np.amax(matrix.tensor, axis=1, keepdims=True)
        matrix_exp = np.exp(matrix.tensor - max_of_row)
        matrix_sum = np.sum(matrix_exp, axis=1, keepdims=True)
        result = matrix_exp / matrix_sum

        return result

    def __setitem__(self, key, val):
        self.tensor[key] = [val for i in self.tensor if i in key]

    def __getitem__(self, val) -> np.ndarray:
        return self.tensor[val]
    
    # Autograd Engine

    # As of now I dont see any need for topological sort as others have used it. 
    # The save_for_backward list contians the nodes already ordered since we build
    # the neural network architecture sequentially.
    def backward(self):
        backward_list.append(self)
        for node in reversed(backward_list):
            node._backward() 

