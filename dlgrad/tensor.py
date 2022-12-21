from __future__ import annotations
import numpy as np
from .graph import draw_graph 


class Tensor:
    def __init__(self, data: np.ndarray):
        self.tensor: np.ndarray = data 

    # Generators 
    @classmethod
    def uniform(cls, shape):
        # The new way 
        return cls(np.random.default_rng().uniform(0, 1, shape))

    # UnaryOps
    def T(self):
        a = Tensor(self.tensor.T)
        return Tensor(self.tensor.T)
    
    # BinaryOps
    def add(self: Tensor, other: Tensor):
        output = Tensor(self.tensor + other.tensor)
        draw_graph(
            'add',
            (output.tensor.shape, 'output'),
            (self.tensor.shape, 'input 1'), 
            (other.tensor.shape, 'input 2')
        )
        return output

    def matmul(self: Tensor, other: Tensor):
        output = Tensor(self.tensor @ other.tensor)
        draw_graph(
            'matmul',
            (output.tensor.shape, 'output'),
            (self.tensor.shape, 'input 1'),
            (other.tensor.shape, 'input 2')
        )
        return output 

    # Others
    def size(self):
        return f"Tensor.size({list(self.tensor.shape)})"

    # def shape(self):
        # return self.tensor.shape

    def __repr__(self) -> str:
        return f"Tensor({self.tensor})"



    