from __future__ import annotations
import numpy as np
from .graph import add_nodes


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
        return Tensor(self.tensor.T)

    
    # BinaryOps

    def add(self: Tensor, other: Tensor):
        output = Tensor(self.tensor + other.tensor)
        add_nodes('add', output, self, other)

    
    def matmul(self: Tensor, other: Tensor):
        output = Tensor(self.tensor @ other.tensor)
        add_nodes('matmul', output, self, other)
        return other

    # Others
    
    def size(self):
        return f"Tensor.size({list(self.tensor.shape)})"

    def shape(self):
        return self.tensor.shape

    def __repr__(self) -> str:
        return f"Tensor({self.tensor})"



    