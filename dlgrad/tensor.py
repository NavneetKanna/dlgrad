from __future__ import annotations
import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray) -> None:
        self.tensor: Tensor = data 

    @classmethod
    def arange(cls, *shape, **kargs): 
        return cls(np.arange(shape, **kargs))

    @classmethod
    def eye(cls, shape, **kargs): 
        return cls(np.eye(shape, **kargs))
    
    @classmethod
    def empty(cls, *shape, **kargs): 
        return cls(np.empty(shape, **kargs))

    @classmethod
    def zeros(cls, *shape, **kargs): 
       ...

    @classmethod
    def ones(cls, *shape, **kargs): 
       ... 

    def T(self):
        return self.tensor.T()

    def matmul(self: Tensor, other: Tensor):
        other = self.tensor @ other.tensor 
        return Tensor(other)

    def size(self):
        return f"Tensor.size({list(self.tensor.shape)})"

    def __repr__(self) -> str:
        return f"Tensor({self.tensor})"



    