from __future__ import annotations
import numpy as np
from .graph import add_nodes


class Tensor:
    idx = 0
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
        # print(type(a))
        # print(a.tensor.shape)
        # print(a.tensor.T.shape)
        return Tensor(self.tensor.T)

    
    # BinaryOps

    def add(self: Tensor, other: Tensor):
        output = Tensor(self.tensor + other.tensor)
        add_nodes(f'add_{Tensor.idx}', output.tensor.shape, self.tensor.shape, other.tensor.shape)
        Tensor.idx += 1
        return output

    
    def matmul(self: Tensor, other: Tensor):
        output = Tensor(self.tensor @ other.tensor)
        print(f"idx {Tensor.idx}")
        add_nodes(f'matmul_{Tensor.idx}', output.tensor.shape, self.tensor.shape, other.tensor.shape)
        Tensor.idx += 1
        return output 

    # Others
    
    def size(self):
        return f"Tensor.size({list(self.tensor.shape)})"

    # def shape(self):
        # return self.tensor.shape

    def __repr__(self) -> str:
        return f"Tensor({self.tensor})"



    