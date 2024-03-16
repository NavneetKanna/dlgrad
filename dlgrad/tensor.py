"""
1) how to detect a left tensor in computational graph ?
2) AssertionError: backward can only be called for scalar tensors
3) permutation
"""

from __future__ import annotations
from typing import Union
from dlgrad.c_interface import c_rand_buffer
from dlgrad.helpers import ShapeError, calculate_stride, calculate_nchw_offset
import ctypes
import atexit
import numpy as np

class Tensor:
    """
    as of now i will go with NCHW, but later if required i will change to channel last
    """
    # __slots__ = "grad"

    def __init__(self, data: Union[list, int, float], view: bool = False, _offset: int = 0, _len: int = None, _shape: tuple = None):
        """
        _len = Number of elements, for ex, (4, 2) -> 8, (2, 3, 4) -> 24
        """
        # self.device = ...
        # self.stride = ...
        # self.grad = ...   

        if isinstance(data, Union[int, float]): 
            self._data = data
            self._len = len(data)
        if isinstance(data, list): 
            self._data = data
            self._len = len(data)
        # TODO: An enum for this ?
        if isinstance(data, ctypes.POINTER(ctypes.c_float)): 
            self._data = data
            self._offset = _offset
            self._len = _len
            # TODO: Add enum/dtype, here 4 = 4 bytes for float32 
            self._total_bytes = _len * 4
            self._shape = _shape
            self._strides = calculate_stride(_shape)
            self.ndim = len(_shape)

        if not view: atexit.register(self.cleanup)

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

        return Tensor(c_rand_buffer._create(size), _offset=0, _len=size, _shape=shape)
    
    def numpy(self):
        sd = ctypes.addressof(self._data.contents) + self._offset * ctypes.sizeof(ctypes.c_float)
        ptr = (ctypes.c_float * self._len).from_address(sd)
        data = np.frombuffer(ptr, count=self._len, dtype=np.float32).reshape(self._shape)
        print(data)

    # TODO: If its a view add a bool = true
    # def __repr__(self) -> str:
    #     return f"{self._data}"

    def cleanup(self): 
        if isinstance(self._data, ctypes.POINTER(ctypes.c_float)): c_rand_buffer._free(self._data)  
    
    def __getitem__(self, indices): 
        # TODO: all int, slices

        if type(indices) == int:
                # if row >= self._shape[0]: raise IndexError 
                offset = calculate_nchw_offset(h=indices, H=self._strides[0]) 
                size = self._strides[0]
                return Tensor(self._data, view=True, _offset=offset, _len=size, _shape=(size,))
        
        if self.ndim == 2:
            if type(indices) == tuple:
                h, w = indices
                offset = calculate_nchw_offset(h=h, w=w, H=self._strides[0]) 
                # if row >= self._shape[0] or col >= self._shape[1]: raise IndexError 
                size = 1
                return Tensor(self._data, view=True, _offset=offset, _len=size, _shape=(size,))
        
        elif self.ndim == 3:
            if type(indices) == tuple:
                length = len(indices)
                if length == 3:
                    c, h, w = indices
                    offset = calculate_nchw_offset(c=c, h=h, w=w, C=self._strides[0], H=self._strides[1]) 
                    size = 1
                    return Tensor(self._data, view=True, _offset=offset, _len=size, _shape=(size,))
                elif length == 2:
                    c, h = indices
                    offset = calculate_nchw_offset(c=c, h=h, C=self._strides[0], H=self._strides[1]) 
                    size = self._strides[1] 
                    return Tensor(self._data, view=True, _offset=offset, _len=size, _shape=(size,))
        
        elif self.ndim == 4:
            if type(indices) == tuple:
                length = len(indices)
                if length == 4:
                    n, c, h, w = indices
                    offset = calculate_nchw_offset(n=n, c=c, h=h, w=w, N=self._strides[0], C=self._strides[1], H=self._strides[2]) 
                    size = 1
                    return Tensor(self._data, view=True, _offset=offset, _len=size, _shape=(size,))
                elif length == 3:
                    n, c, h = indices
                    offset = calculate_nchw_offset(n=n, c=c, h=h, N=self._strides[0], C=self._strides[1], H=self._strides[2]) 
                    size = self._strides[2] 
                    return Tensor(self._data, view=True, _offset=offset, _len=size, _shape=(size,))
                elif length == 2:
                    n, c  = indices
                    offset = calculate_nchw_offset(n=n, c=c, N=self._strides[0], C=self._strides[1]) 
                    size = self._strides[1] 
                    return Tensor(self._data, view=True, _offset=offset, _len=size, _shape=(size,))
                


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