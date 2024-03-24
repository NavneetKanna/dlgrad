"""
1) how to detect a left tensor in computational graph ?
2) AssertionError: backward can only be called for scalar tensors
3) permutation
4) kernel dispatch
"""

from __future__ import annotations
from typing import Union
from dlgrad.c_interface import c_rand_buffer, c_add
from dlgrad.helpers import ShapeError, IndexError, calculate_stride, calculate_nchw_offset
import ctypes
import atexit
import numpy as np
import platform
import inspect
from dlgrad import c_interface

class Tensor:
    """
    as of now i will go with NCHW, but later if required i will change to channel last
    """
    # __slots__ = "grad"

    # TODO: is it a good programming practice to have many args ?
    def __init__(
            self,
            data: Union[list, int, float],
            requires_grad = True,
            device: str = "",
            view: bool = False,
            _offset: int = 0,
            _len: int = 0,
            _shape: tuple = ()
        ):
        """
        _len = Number of elements, for ex, (4, 2) -> 8, (2, 3, 4) -> 24
        """

        # TODO: add dtype
        self.requires_grad = True

        if device is None and platform.system() == 'Darwin' and platform.processor() == 'arm':
                self._device = 'metal'
        else: self._device = device

        if isinstance(data, Union[int, float]):
            self._data = data
            self._len = 1
            self._view = view
            self._contig = False
        # TODO: Convert this to c array
        if isinstance(data, list):
            self._data = data
            self._len = len(data)
            self._view = view
            self._contig = False
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
            self._view = view
            self._contig = True

        if not view: atexit.register(self.cleanup)

    def numpy(self):
        sd = ctypes.addressof(self._data.contents) + self._offset * ctypes.sizeof(ctypes.c_float)
        ptr = (ctypes.c_float * self._len).from_address(sd)
        data = np.frombuffer(ptr, count=self._len, dtype=np.float32).reshape(self._shape)
        print(data)

    def __repr__(self) -> str:
        return f"Tensor <contig: {self._contig} view:{self._view} device: {self._device}>"

    def cleanup(self):
        print(type(self._data))
        c_rand_buffer._free(self._data)
        # c_add._free(self._data)

        if isinstance(self._data, ctypes.POINTER(ctypes.c_float)): 
            classes = inspect.getmembers(c_interface, inspect.isclass)

            for name, cls in classes:
                if hasattr(cls, '_free'):
                    print(cls)
                    # cls.__free()


    def __getitem__(self, indices):
        # TODO: all int, slices

        if type(indices) == int:
                if indices > self._shape[0]:
                    raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")

                offset = calculate_nchw_offset(h=indices, H=self._strides[0])
                size = self._strides[0]
                return Tensor(self._data, device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))

        if self.ndim == 2:
            if type(indices) == tuple:
                h, w = indices
                if h > self._shape[0]:
                    raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                if w > self._shape[1]:
                    raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape}")

                offset = calculate_nchw_offset(h=h, w=w, H=self._strides[0])
                size = 1
                return Tensor(self._data, device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))

        elif self.ndim == 3:
            if type(indices) == tuple:
                length = len(indices)
                if length == 3:
                    c, h, w = indices
                    if c > self._shape[0]:
                        raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                    if h > self._shape[1]:
                        raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape}")
                    if w > self._shape[2]:
                        raise IndexError(f"index {indices} > {self._shape[2]} of {self._shape}")

                    offset = calculate_nchw_offset(c=c, h=h, w=w, C=self._strides[0], H=self._strides[1])
                    size = 1
                    return Tensor(self._data, device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))
                elif length == 2:
                    c, h = indices
                    if c > self._shape[0]:
                        raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                    if h > self._shape[1]:
                        raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape} ")

                    offset = calculate_nchw_offset(c=c, h=h, C=self._strides[0], H=self._strides[1])
                    size = self._strides[1]
                    return Tensor(self._data, device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))

        elif self.ndim == 4:
            if type(indices) == tuple:
                length = len(indices)
                if length == 4:
                    n, c, h, w = indices
                    if n > self._shape[0]:
                        raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                    if c > self._shape[1]:
                        raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape}")
                    if h > self._shape[2]:
                        raise IndexError(f"index {indices} > {self._shape[2]} of {self._shape}")
                    if w > self._shape[3]:
                        raise IndexError(f"index {indices} > {self._shape[3]} of {self._shape}")

                    offset = calculate_nchw_offset(n=n, c=c, h=h, w=w, N=self._strides[0], C=self._strides[1], H=self._strides[2])
                    size = 1
                    return Tensor(self._data, device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))
                elif length == 3:
                    n, c, h = indices
                    if n > self._shape[0]:
                        raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                    if w > self._shape[1]:
                        raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape}")
                    if h > self._shape[2]:
                        raise IndexError(f"index {indices} > {self._shape[2]} of {self._shape}")

                    offset = calculate_nchw_offset(n=n, c=c, h=h, N=self._strides[0], C=self._strides[1], H=self._strides[2])
                    size = self._strides[2]
                    return Tensor(self._data, device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))
                elif length == 2:
                    n, c  = indices
                    if n > self._shape[0]:
                        raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                    if c > self._shape[1]:
                        raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape}")

                    offset = calculate_nchw_offset(n=n, c=c, N=self._strides[0], C=self._strides[1])
                    size = self._strides[1]
                    return Tensor(self._data, device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))


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

        return Tensor(c_rand_buffer._create(size), _offset=0, _len=size, _shape=shape)

    # TODO: where is broadcasting used ?
    # TODO: support +
    # ***** BinaryOps *****
    @staticmethod
    def add(x: Tensor, y: Tensor):
        assert x._shape == y._shape, f"{x._shape} and {y._shape} does not match"

        def _backward(): pass

        # TODO: assert device
        return Tensor(c_add._add(x._data, y._data, x._len), device=x._device, _len=x._len, _shape=x._shape)

"""
https://arxiv.org/abs/1502.05767

https://cs231n.github.io/optimization-2/

https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

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
