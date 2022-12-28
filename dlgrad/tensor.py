from __future__ import annotations
import numpy as np
from .graph import draw_graph 

class Tensor:
    save_for_backward = []
    weight_parameters = []
    bias_parameters = []
    parameters = []

    def __init__(self, data: np.ndarray):
        self.tensor: np.ndarray = data 
        # Check first if it has tensor attribute
        self.shape = self.tensor.shape
        self._backward = lambda: None
        self.grad = np.zeros(self.shape)

    @classmethod
    def uniform(cls, shape: tuple):
        # The new way 
        return cls(np.random.default_rng().uniform(-1, 1, shape))

    @classmethod
    def zeros(cls, shape: tuple):
        return cls(np.zeros(shape))

    # UnaryOps
    # def T(self):
    #     return Tensor(self.tensor.T)
    
    # BinaryOps
    def add(self: Tensor, other: Tensor, flag):
        output = Tensor(self.tensor + other.tensor)
        Tensor.save_for_backward.extend((self, other))

        if flag:
            draw_graph(
                'add',
                (output.tensor.shape, 'output', output.grad.shape),
                (self.tensor.shape, 'input 1', self.grad.shape), 
                (other.tensor.shape, 'input 2', other.grad.shape)
            )
            
        def backward():
            self.grad =  output.grad
            other.grad =  np.sum(output.grad, axis=0, keepdims=True)
        
        output._backward = backward

        return output

    def sub(self, other):
        return Tensor(self.tensor-other.tensor)

    def matmul(self: Tensor, other: Tensor, flag):
        output = Tensor(self.tensor @ other.tensor.T)
        Tensor.save_for_backward.extend((self, other))

        if flag:
            draw_graph(
                'matmul',
                (output.tensor.shape, 'output', output.grad.shape),
                (self.tensor.shape, 'input 1', self.grad.shape),
                (other.tensor.shape, 'input 2', other.grad.shape)
            )

        def backward():
            self.grad =  (output.grad @ other.tensor)
            other.grad =  (self.tensor.T @ output.grad).T

        output._backward = backward

        return output 

    def __repr__(self) -> str:
        return f"Tensor({self.tensor})"

    def __getitem__(self, val):
        return Tensor(self.tensor[val])

    def get_weight_parameters():
        return Tensor.weight_parameters 
    
    def get_bias_parameters():
        return Tensor.bias_parameters
    
    def get_parameters():
        return Tensor.parameters

    # Autograd Engine

    # As of now I dont see any need for topological sort as others have used it. 
    # The save_for_backward list contians the nodes already ordered since we build
    # the neural network architecture sequentially.
    def backward(self):
        Tensor.save_for_backward.append(self)
        for node in reversed(Tensor.save_for_backward):
            node._backward() 
