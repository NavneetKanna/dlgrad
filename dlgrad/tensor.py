from __future__ import annotations
import numpy as np
from .graph import CG

class Tensor:
    # len = sum((no of layers*2)) + 3 
    save_for_backward = []
    parameters = []
    cg = CG()

    def __init__(self, data: np.ndarray):
        self.tensor: np.ndarray = data 
        self.shape = self.tensor.shape
        self._backward = lambda: None
        self.grad = np.zeros(self.shape)

    @classmethod
    def uniform(cls, shape: tuple):
        return cls(np.random.default_rng().uniform(-1, 1, shape))

    @classmethod
    def zeros(cls, shape: tuple):
        return cls(np.zeros(shape))

    def max(data, axis):
        return np.max(data.tensor, axis=axis)
    
    # Ops
    def add(self: Tensor, other: Tensor):
        output = Tensor(self.tensor + other.tensor)
        Tensor.save_for_backward.extend((self, other))

        if not CG.stop_processing: CG.add_nodes('add', output.tensor, self.tensor, other.tensor)

        def backward():
            self.grad = output.grad
            other.grad = np.sum(output.grad, axis=0, keepdims=True)
        
        output._backward = backward

        return output

    def matmul(self: Tensor, other: Tensor):
        output = Tensor(self.tensor @ other.tensor.T)
        Tensor.save_for_backward.extend((self, other))

        if not CG.stop_processing: CG.add_nodes('matmul', output.tensor, self.tensor, other.tensor)

        def backward():
            self.grad = (output.grad @ other.tensor)
            other.grad = (self.tensor.T @ output.grad).T

        output._backward = backward

        return output 

    def __setitem__(self, key, val):
        self.tensor[key] = [val for i in self.tensor if i in key]

    def __getitem__(self, val) -> np.ndarray:
        return self.tensor[val]

    def get_parameters():
        return Tensor.parameters
    
    @staticmethod
    def zero_grad():
        for parameters in Tensor.save_for_backward:
            parameters.grad = np.zeros(parameters.shape) 
        Tensor.save_for_backward.clear()

    # Update the weights and biases
    @staticmethod
    def step(lr):
        for parameters in Tensor.get_parameters():
            parameters.tensor = parameters.tensor - (lr*parameters.grad)
        
    # Autograd Engine

    # As of now I dont see any need for topological sort as others have used it. 
    # The save_for_backward list contians the nodes already ordered since we build
    # the neural network architecture sequentially.
    def backward(self):
        Tensor.save_for_backward.append(self)
        for node in reversed(Tensor.save_for_backward):
            node._backward() 

