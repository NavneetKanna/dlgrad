"""
1) how to detect a left tensor in computational graph ?
2) AssertionError: backward can only be called for scalar tensors
3) detect if list is n dim
4)  @property

"""

from __future__ import annotations
from typing import Union
from dlgrad.buffer import Buffer

class Tensor:
    # __slots__ = "grad"

    def __init__(self, data: Union[list, int, float]):
        # self.device = ...
        # self.stride = ...
        # self.grad = ...
        self.data = None
    
        if isinstance(data, Union[int, float]): self.data = Buffer.create_scalar_buffer(data)    
        if isinstance(data, list): self.data = Buffer.create_list_buffer(data)