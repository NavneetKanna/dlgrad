"""
1) how to detect a left tensor in computational graph ?
2) AssertionError: backward can only be called for scalar tensors
"""

from __future__ import annotations
from typing import Union
from dlgrad.c_interface import c_rand_buffer
from dlgrad.helpers import ShapeError, calculate_stride
import ctypes
import atexit
import numpy as np

class Tensor:
    # __slots__ = "grad"

    def __init__(self, data: Union[list, int, float], _total_bytes: int=None, _shape: tuple = None):
        # self.device = ...
        # self.stride = ...
        # self.grad = ...   

        if isinstance(data, Union[int, float]): 
            self._data = data
            self.len = len(data)
        if isinstance(data, list): 
            self._data = data
            self.len = len(data)
        # TODO: An enum for this ?
        if isinstance(data, ctypes.POINTER(ctypes.c_float)): 
            self._data = data
            self._len = _total_bytes
            self._shape = _shape
            self._strides = calculate_stride(_shape)
            self.ndim = len(_shape)

        atexit.register(self.cleanup)

    # ***** DCOPS (data creation ops) *****
    @staticmethod
    def rand(*shape) -> Tensor: 
        # TODO: ((((?))))
        if isinstance(shape[0], tuple): shape = shape[0]
        if isinstance(shape[0], list): shape = tuple(*shape)

        if len(shape) > 4: raise ShapeError("dlgrad only supports upto 4 dim")
        if isinstance(shape[0], list): raise ShapeError("Multi-dim list cannot be passed as shape")
        if not all(isinstance(item, int) for item in shape): raise ShapeError("Only ints can be passed to shape")
        size = 1
        for i in shape: size *= i
        
        return Tensor(c_rand_buffer._create(size), _total_bytes=size, _shape=shape)
    
    def numpy(self):
        ptr = ctypes.cast(self._data, ctypes.POINTER(ctypes.c_float * self._len))
        data = np.frombuffer(ptr.contents, dtype=np.float32).reshape(self._shape)
        print(data)

    # def __repr__(self) -> str:
    #     return f"{self._data}"

    def cleanup(self): 
        if isinstance(self._data, ctypes.POINTER(ctypes.c_float)): c_rand_buffer._free(self._data)  
    
    def __getitem__(self, indices): 
        # TODO: all int
        """
        start_idx = data(idx[dim]*stride[dim] till stride[dim])
        
        """
        # TODO: slices
        print(type(self))
        print(indices)
        print(type(indices))

        if self.ndim == 2:
            row = ...
            col = ...
            if type(indices) == int:
                row = indices
                if row > self.ndim: raise IndexError 
                offset = row * self._strides[0] + col
                
            elif type(indices) == tuple:
                pass

            offset = row * self._shape[1] + col

      



        


"""
https://stackoverflow.com/questions/7343833/srand-why-call-it-only-once

for rand
--------
read the buffer into 1D byte array
now shape the array as you please

ptr = ctypes.cast(a._data, ctypes.POINTER(ctypes.c_float * 2))
np.asarray(ptr.contents)
np.frombuffer(ptr.contents, dtype=np.float32)

np.ctypeslib.as_array(a._data, shape=(1, 2))
"""