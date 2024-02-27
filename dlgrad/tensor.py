"""
1) how to detect a left tensor in computational graph ?
2) AssertionError: backward can only be called for scalar tensors
3) detect if list is n dim
4)  @property
5) list should be homogeneous
"""

from __future__ import annotations
from typing import Union
from dlgrad.buffer import Buffer
from dlgrad.helpers import ShapeError

class Tensor:
    # __slots__ = "grad"

    def __init__(self, data: Union[list, int, float]):
        # self.device = ...
        # self.stride = ...
        # self.grad = ...
       
        self.data = None # TODO: change repr when called
    
        if isinstance(data, Union[int, float]): self.data = Buffer.create_scalar_buffer(data)    
        if isinstance(data, list): self.data = Buffer.create_list_buffer(data)

    # ***** DCOPS (data creation ops) *****
    @staticmethod
    def rand(*shape): 
        if isinstance(shape[0], tuple): shape = shape[0]
        if isinstance(shape[0], list): shape = tuple(*shape)

        if not all(isinstance(item, int) for item in shape): raise ShapeError("Only ints can be passed to shape")
        if len(shape) > 4: raise ShapeError("dlgrad only supports upto 4 dim")
        if isinstance(shape[0], list): raise ShapeError("Multi-dim list cannot be passed as shape")

        Buffer.create_rand_buffer(shape)