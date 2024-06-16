from __future__ import annotations
from typing import Union, Optional
from dlgrad.helpers import ShapeError, IndexError, calculate_stride, calculate_nchw_offset, BinaryOps, UnaryOps, Device, BroadcastHelper
import ctypes
import atexit
import numpy as np
# import platform
from dlgrad.dtype import dtypes
from dlgrad.buffer import Buffer
import warnings
from dlgrad.dispatch import Dispatcher
import math
from dataclasses import dataclass

@dataclass
class TensorProperties:
    view: bool = False
    offset: int = 0
    numel: int = 0
    shape: tuple = (1,)
    ndim: int = 0
    strides: tuple = (1,)
    contig: bool = True

class Tensor:
    # __slots__ = "grad"

    # TODO: is it a good programming practice to have many args ?
    def __init__(
            self, data: Union[list, int, Buffer], requires_grad = True, device: Device = Device.CPU, 
            dtype: Optional[dtypes] = None, properties: TensorProperties = TensorProperties()
        ):
        self.requires_grad = requires_grad

        # if device is None and platform.system() == 'Darwin' and platform.processor() == 'arm': 
        #     self._device = 'metal'
        # else: 
        #     self._device = device
        self._device = device

        if isinstance(data, Union[int, float]):
            self._data = data
            self._len = 1
            self._view = view
            self._contig = False
            self.dtype = dtype if dtype else dtypes.from_py(data)

        # TODO: Convert this to c array
        if isinstance(data, list):
            self._data = data
            self._len = len(data)
            self._view = view
            self._contig = False

        if isinstance(data, Buffer):
            self._data = data.data_buffer
            self.properties = properties
            self._dtype = dtype

        if not view and isinstance(data, Buffer): 
            atexit.register(self.cleanup)

    def numpy(self):
        sd = ctypes.addressof(self._data.contents) + self._offset * ctypes.sizeof(ctypes.c_float)
        ptr = (ctypes.c_float * self._len).from_address(sd)
        data = np.frombuffer(ptr, count=self._len, dtype=np.float32).reshape(self._shape)
        print(data)

    @staticmethod
    def _broadcast(x: Tensor, y: Tensor):
        shape1 = x._shape
        shape2 = y._shape

        if x._ndim > 2 or y._ndim > 2 and shape1 != shape2:
            print("Dlgrad does not support broadcasting for dims greater than 2")
        
        output_shape = []
        
        shape1 = shape1[::-1]
        shape2 = shape2[::-1]

        for i in range(max(len(shape1), len(shape2))):
            dim1 = shape1[i] if i < len(shape1) else 1
            dim2 = shape2[i] if i < len(shape2) else 1
            if dim1 == 1 or dim2 == 1 or dim1 == dim2:
                output_shape.append(max(dim1, dim2))
            else:
                # TODO: Add error here
                print("Shapes are not compatible for broadcasting")
        
        return tuple(output_shape[::-1])

    def __repr__(self) -> str:
        return f"Tensor <contig: {self._contig} view:{self._view} device: {self._device}>"

    def linear(self, weight: Tensor, bias: Tensor) -> Tensor:
        # self*weight.T + bias
        return Tensor.add(Tensor.matmul(self, Tensor.transpose(weight)), bias)

    # TODO: maybe do a check for data before calling free ?
    def cleanup(self): 
        Buffer.free(self._data)

    def __getitem__(self, indices):
        # TODO: all int, slices
        # NOTE: dlgrad is NCHW

        if isinstance(indices, int):
            if indices > self._shape[0]:
                raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")

            offset = calculate_nchw_offset(h=indices, H=self._strides[0])
            size = self._strides[0]
            tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), strides=calculate_stride(size), contig=True)
            return Tensor(Buffer(self._data), device=self._device, properties=tp)

        if self._ndim == 2:
            if type(indices) == tuple:
                h, w = indices
                if h > self._shape[0]:
                    raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                if w > self._shape[1]:
                    raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape}")

                offset = calculate_nchw_offset(h=h, w=w, H=self._strides[0])
                size = 1
                tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), strides=calculate_stride(size), contig=True)
                return Tensor(Buffer(self._data), device=self._device, properties=tp)

        elif self._ndim == 3:
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
                    tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), strides=calculate_stride(size), contig=True)
                    return Tensor(Buffer(self._data), device=self._device, properties=tp)
                elif length == 2:
                    c, h = indices
                    if c > self._shape[0]:
                        raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                    if h > self._shape[1]:
                        raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape} ")

                    offset = calculate_nchw_offset(c=c, h=h, C=self._strides[0], H=self._strides[1])
                    size = self._strides[1]
                    tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), strides=calculate_stride(size), contig=True)
                    return Tensor(Buffer(self._data), device=self._device, properties=tp)

        elif self._ndim == 4:
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
                    tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), strides=calculate_stride(size), contig=True)
                    return Tensor(Buffer(self._data), device=self._device, properties=tp)
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
                    tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), strides=calculate_stride(size), contig=True)
                    return Tensor(Buffer(self._data), device=self._device, properties=tp)
                elif length == 2:
                    n, c  = indices
                    if n > self._shape[0]:
                        raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                    if c > self._shape[1]:
                        raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape}")

                    offset = calculate_nchw_offset(n=n, c=c, N=self._strides[0], C=self._strides[1])
                    size = self._strides[1]
                    tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), strides=calculate_stride(size), contig=True)
                    return Tensor(Buffer(self._data), device=self._device, properties=tp)

    # ***** BufferOps *****
    # BufferOps as of now uses only cpu to generate data
    @staticmethod
    def rand(*shape, low = 0.0, high = 1.0, device: Device = Device.CPU, dtype: Optional[dtypes] = dtypes.float32) -> Tensor:
        if device != Device.CPU:
            warnings.warn("Currently BufferOps are only created on CPU.")

        if dtype != dtypes.float32:
            warnings.warn("Currently dlgrad only supports float32, but more dtypes coming in future. Creating data with dtype=float32.")

        if isinstance(shape[0], tuple): 
            shape = shape[0]
        if isinstance(shape[0], list): 
            shape = tuple(*shape)

        if len(shape) > 4: 
            raise ShapeError("dlgrad only supports upto 4 dim")
        if isinstance(shape[0], list): 
            raise ShapeError("Multi-dim list cannot be passed as shape")
        if not all(isinstance(item, int) for item in shape): 
            raise ShapeError("Only ints can be passed to shape")
        out_len = 1
        for i in shape: 
            out_len *= i

        tp = TensorProperties(view=False, offset=0, numel=out_len, shape=shape, ndim=len(shape), strides=calculate_stride(shape), contig=True)
        return Tensor(Buffer.uniform(out_len, low, high), _offset=0, device=device, dtype=dtype, properties=tp)

    @staticmethod
    def kaiming_uniform(*shape, device = Device.CPU, dtype = dtypes.float32):
        if isinstance(shape[0], tuple): 
            shape = shape[0]
        if isinstance(shape[0], list): 
            shape = tuple(*shape)

        size = 1
        for i in shape: 
            size *= i

        a = math.sqrt(5)
        gain = math.sqrt(2 / (1 + a**2))
        std = gain / math.sqrt(shape[1])
        bound = math.sqrt(3) * std
        
        return Tensor(Buffer.uniform(size, low=-bound, high=bound), _offset=0, device=device, dtype=dtype, _len=size, _shape=shape)
    
    # ***** UnaryOps ****
    # TODO: What to do if i want to call x.transpose() ?
    @staticmethod
    def transpose(x: Tensor):
        return Tensor(Dispatcher.dispatch(x, ops=UnaryOps.TRANSPOSE), device=x._device, _len=x._len, _shape=x._shape[::-1], view=False)

    # ***** ElementwiseOps *****
    # TODO: Dont like the way dispatch is getting called
    @staticmethod
    def add(x: Tensor, y: Tensor) -> Tensor:
        assert x._device == y._device, f"{x._device} and {y._device} does not match"

        out_shape = Tensor._broadcast(x, y)
        out_len = 1
        for i in out_shape:
            out_len *= i

        BroadcastHelper.out_len = out_len

        def _backward(): pass

        return Tensor(Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.ADD), device=x._device, dtype=x._dtype, _len=out_len, _shape=out_shape, view=False)
    
    # ***** BinaryOps *****
    @staticmethod
    def matmul(x: Tensor, y: Tensor) -> Tensor:
        # TODO: Check dtype as well
        assert x._shape[-1] == y._shape[0], f"{x._shape} and {y._shape} does not match"
        assert x._device == y._device, f"{x._device} and {y._device} does not match"

        def _backward(): pass

        # TODO: How do i ensure data is of same dtype
        return Tensor(Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.MATMUL), device=x._device, dtype=x._dtype, _len=x._shape[0]*y._shape[1], _shape=(x._shape[0], y._shape[1]), view=False)
