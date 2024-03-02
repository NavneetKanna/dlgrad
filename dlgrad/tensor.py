"""
1) how to detect a left tensor in computational graph ?
2) AssertionError: backward can only be called for scalar tensors
3) detect if list is n dim
4)  @property
5) list should be homogeneous
6) clang -rand
"""

from __future__ import annotations
from typing import Union
from dlgrad.c_interface import c_rand_buffer
from dlgrad.helpers import ShapeError
import ctypes
import weakref
import atexit

class Tensor:
    # __slots__ = "grad"

    def __init__(self, data: Union[list, int, float]):
        # self.device = ...
        # self.stride = ...
        # self.grad = ...   

        if isinstance(data, Union[int, float]): self._data = data    
        if isinstance(data, list): self._data = data
        if isinstance(data, ctypes.POINTER(ctypes.c_float)): self._data = data

        atexit.register(self.cleanup)

    # ***** DCOPS (data creation ops) *****
    @staticmethod
    def rand(*shape) -> Tensor: 
        if isinstance(shape[0], tuple): shape = shape[0]
        if isinstance(shape[0], list): shape = tuple(*shape)

        if len(shape) > 4: raise ShapeError("dlgrad only supports upto 4 dim")
        if isinstance(shape[0], list): raise ShapeError("Multi-dim list cannot be passed as shape")
        if not all(isinstance(item, int) for item in shape): raise ShapeError("Only ints can be passed to shape")
        size = 1
        for i in shape: size *= i
        # return Tensor(Buffer._create_rand_buffer(shape))
        return Tensor(c_rand_buffer._create(size))
    
    def __repr__(self) -> str:
        return f"{self._data}"

    def cleanup(self): 
        if isinstance(self._data, ctypes.POINTER(ctypes.c_float)): c_rand_buffer._free(self._data)  
    # TODO: Have index func 

"""
https://stackoverflow.com/questions/7343833/srand-why-call-it-only-once

"""