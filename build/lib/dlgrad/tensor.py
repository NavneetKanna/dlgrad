from __future__ import annotations
import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray):
        self.tensor: np.ndarray = data 

    @classmethod
    def uniform(cls, shape):
        return cls(np.random.default_rng().uniform(0, 1, shape))

    def T(self):
        return Tensor(self.tensor.T)

    def matmul(self: Tensor, other: Tensor):
        other = self.tensor @ other.tensor 
        return Tensor(other)

    def size(self):
        return f"Tensor.size({list(self.tensor.shape)})"

    def shape(self):
        return self.tensor.shape

    def __repr__(self) -> str:
        return f"Tensor({self.tensor})"



    