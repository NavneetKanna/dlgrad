from __future__ import annotations
import numpy as np
from .graph import draw_graph 


class Tensor:
    save_for_backward = []
    parameters = []

    def __init__(self, data: np.ndarray):
        self.tensor: np.ndarray = data 
        # Check first if it has tensor attribute
        self.shape = self.tensor.shape
        self._backward = lambda: None
        self.grad = np.zeros(self.shape)

    def save_parents(self, parents: tuple=None):
        Tensor.save_for_backward.append(parents)

    @classmethod
    def uniform(cls, shape: tuple):
        # The new way 
        return cls(np.random.default_rng().uniform(0, 1, shape))

    @classmethod
    def zeros(cls, shape: tuple):
        return cls(np.zeros(shape))

    # UnaryOps
    def T(self):
        a = Tensor(self.tensor.T)
        return Tensor(self.tensor.T)
    
    # BinaryOps
    def add(self: Tensor, other: Tensor):
        output = Tensor(self.tensor + other.tensor)
        Tensor.save_for_backward.append(self, other)

        draw_graph(
            'add',
            (output.tensor.shape, 'output'),
            (self.tensor.shape, 'input 1'), 
            (other.tensor.shape, 'input 2')
        )
        
        def backward():
            self.grad += np.identity(self.shape[0]) @ output.grad
            other.grad += np.identity(other.shape[0]) @ output.grad
        
        output.backward = backward

        return output

    def matmul(self: Tensor, other: Tensor):
        output = Tensor(self.tensor @ other.tensor)
        Tensor.save_for_backward.append(self, other)

        draw_graph(
            'matmul',
            (output.tensor.shape, 'output'),
            (self.tensor.shape, 'input 1'),
            (other.tensor.shape, 'input 2')
        )

        def backward():
            self.grad += output.grad @ other.tensor.T
            other.grad += self.tensor.T @ output.grad

        output.backward = backward

        return output 

    # Others
    # def shape(self):
    #     return self.tensor.shape

    def __repr__(self) -> str:
        return f"Tensor({self.tensor})"

    def __getitem__(self, val):
        return Tensor(self.tensor[val])

    def get_parameters():
        return Tensor.parameters 


    # Autograd Engine

    # As of now I dont see any need for topological sort as others have used it. 
    # The save_for_backward list contians the nodes already ordered since we build
    # the neural network architecture sequentially.
    def backward(self):
        # Add the last node(the loss) to the list
        Tensor.save_for_backward(self)

        # TODO: Is this necessary ?
        self.grad = np.ones(self.shape)

        for node in reversed(Tensor.save_for_backward):
            node._backward() 




    