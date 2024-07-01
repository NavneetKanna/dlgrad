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
import dlgrad.ops as ops


@dataclass
class TensorProperties:
    view: bool = False
    offset: int = 0
    numel: int = 0
    shape: tuple = ()   
    ndim: int = 0
    stride: tuple = ()
    contig: bool = True



# TODO: Maybe we can load all ctypes files once in the beginning, so that it does not take time to load ?
class Tensor:
    # __slots__ = "grad"

    def __init__(
            self, data: Union[list, int, Buffer], requires_grad = True, device: Device = Device.CPU, 
            dtype: Optional[dtypes] = None, properties: TensorProperties = TensorProperties()
        ):
        self.requires_grad = requires_grad
        self.grad = None
        self.device = device
        self.properties = properties
        self._ctx = None

        if isinstance(data, Union[int, float]):
            self.data = data
            self._len = 1
            self.dtype = dtype if dtype else dtypes.from_py(data)

        # TODO: Convert this to c array
        if isinstance(data, list):
            self.data = data

        if isinstance(data, Buffer):
            self.data = data
            self.dtype = dtype

        if not properties.view and isinstance(data, Buffer): 
            atexit.register(self.cleanup)

    def numpy(self):
        if not isinstance(self.data, Buffer):
            data = np.array(self.data)
            print(data)
        else:
            sd = ctypes.addressof(self.data._buffer.contents) + self.offset * ctypes.sizeof(ctypes.c_float)
            ptr = (ctypes.c_float * self._len).from_address(sd)
            data = np.frombuffer(ptr, count=self._len, dtype=np.float32).reshape(self.shape)
            print(data)

    @staticmethod
    def _broadcast(x: Tensor, y: Tensor):
        shape1 = x.shape
        shape2 = y.shape

        if x.ndim > 2 or y.ndim > 2 and shape1 != shape2:
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

    def linear(self, weight: Tensor, bias: Tensor) -> Tensor:
        # self*weight.T + bias
        return Tensor.add(Tensor.matmul(self, Tensor.transpose(weight)), bias)

    # TODO: maybe do a check for data before calling free ?
    def cleanup(self): 
        Buffer.free(self.data._buffer)

    def __getitem__(self, indices):
        # TODO: all int, slices
        # NOTE: dlgrad is NCHW

        if isinstance(indices, int):
            if indices > self.shape[0]:
                raise IndexError(f"index {indices} > {self.shape[0]} of {self.shape}")

            offset = calculate_nchw_offset(h=indices, H=self.stride[0])
            size = self.stride[0]
            tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), stride=calculate_stride(size), contig=True)
            return Tensor(Buffer(self.data._buffer), device=self.device, properties=tp)

        if self.ndim == 2:
            if type(indices) == tuple:
                h, w = indices
                if h > self.shape[0]:
                    raise IndexError(f"index {indices} > {self.shape[0]} of {self.shape}")
                if w > self.shape[1]:
                    raise IndexError(f"index {indices} > {self.shape[1]} of {self.shape}")

                offset = calculate_nchw_offset(h=h, w=w, H=self.stride[0])
                size = 1
                tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), stride=calculate_stride(size), contig=True)
                return Tensor(Buffer(self.data._buffer), device=self.device, properties=tp)

        elif self.ndim == 3:
            if type(indices) == tuple:
                length = len(indices)
                if length == 3:
                    c, h, w = indices
                    if c > self.shape[0]:
                        raise IndexError(f"index {indices} > {self.shape[0]} of {self.shape}")
                    if h > self.shape[1]:
                        raise IndexError(f"index {indices} > {self.shape[1]} of {self.shape}")
                    if w > self.shape[2]:
                        raise IndexError(f"index {indices} > {self.shape[2]} of {self.shape}")

                    offset = calculate_nchw_offset(c=c, h=h, w=w, C=self.stride[0], H=self.stride[1])
                    size = 1
                    tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), stride=calculate_stride(size), contig=True)
                    return Tensor(Buffer(self.data._buffer), device=self.device, properties=tp)
                elif length == 2:
                    c, h = indices
                    if c > self.shape[0]:
                        raise IndexError(f"index {indices} > {self.shape[0]} of {self.shape}")
                    if h > self.shape[1]:
                        raise IndexError(f"index {indices} > {self.shape[1]} of {self.shape} ")

                    offset = calculate_nchw_offset(c=c, h=h, C=self.stride[0], H=self.stride[1])
                    size = self.stride[1]
                    tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), stride=calculate_stride(size), contig=True)
                    return Tensor(Buffer(self.data._buffer), device=self.device, properties=tp)

        elif self.ndim == 4:
            if type(indices) == tuple:
                length = len(indices)
                if length == 4:
                    n, c, h, w = indices
                    if n > self.shape[0]:
                        raise IndexError(f"index {indices} > {self.shape[0]} of {self.shape}")
                    if c > self.shape[1]:
                        raise IndexError(f"index {indices} > {self.shape[1]} of {self.shape}")
                    if h > self.shape[2]:
                        raise IndexError(f"index {indices} > {self.shape[2]} of {self.shape}")
                    if w > self.shape[3]:
                        raise IndexError(f"index {indices} > {self.shape[3]} of {self.shape}")

                    offset = calculate_nchw_offset(n=n, c=c, h=h, w=w, N=self.stride[0], C=self.stride[1], H=self.stride[2])
                    size = 1
                    tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), stride=calculate_stride(size), contig=True)
                    return Tensor(Buffer(self.data._buffer), device=self.device, properties=tp)
                elif length == 3:
                    n, c, h = indices
                    if n > self.shape[0]:
                        raise IndexError(f"index {indices} > {self.shape[0]} of {self.shape}")
                    if w > self.shape[1]:
                        raise IndexError(f"index {indices} > {self.shape[1]} of {self.shape}")
                    if h > self.shape[2]:
                        raise IndexError(f"index {indices} > {self.shape[2]} of {self.shape}")

                    offset = calculate_nchw_offset(n=n, c=c, h=h, N=self.stride[0], C=self.stride[1], H=self.stride[2])
                    size = self.stride[2]
                    tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), stride=calculate_stride(size), contig=True)
                    return Tensor(Buffer(self.data._buffer), device=self.device, properties=tp)
                elif length == 2:
                    n, c  = indices
                    if n > self.shape[0]:
                        raise IndexError(f"index {indices} > {self.shape[0]} of {self.shape}")
                    if c > self.shape[1]:
                        raise IndexError(f"index {indices} > {self.shape[1]} of {self.shape}")

                    offset = calculate_nchw_offset(n=n, c=c, N=self.stride[0], C=self.stride[1])
                    size = self.stride[1]
                    tp = TensorProperties(view=True, offset=offset, numel=size, shape=(size,), ndim=len(size), stride=calculate_stride(size), contig=True)
                    return Tensor(Buffer(self.data._buffer), device=self.device, properties=tp)

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

        tp = TensorProperties(view=False, offset=0, numel=out_len, shape=shape, ndim=len(shape), stride=calculate_stride(shape), contig=True)
        return Tensor(Buffer.uniform(out_len, low, high), device=device, dtype=dtype, properties=tp)

    @staticmethod
    def kaiming_uniform(*shape, device = Device.CPU, dtype = dtypes.float32):
        if isinstance(shape[0], tuple): 
            shape = shape[0]
        if isinstance(shape[0], list): 
            shape = tuple(*shape)

        out_len = 1
        for i in shape: 
            out_len *= i

        a = math.sqrt(5)
        gain = math.sqrt(2 / (1 + a**2))
        std = gain / math.sqrt(shape[1])
        bound = math.sqrt(3) * std
        
        tp = TensorProperties(view=False, offset=0, numel=out_len, shape=shape, ndim=len(shape), stride=calculate_stride(shape), contig=True)
        return Tensor(Buffer.uniform(out_len, low=-bound, high=bound), device=device, dtype=dtype, properties=tp)
    
    # ***** UnaryOps ****
    # TODO: What to do if i want to call x.transpose() ?
    @staticmethod
    def transpose(x: Tensor):
        tp = TensorProperties(view=False, offset=0, numel=x.numel, shape=x.shape[::-1], ndim=len(x.shape[::-1]), stride=calculate_stride(x.shape[::-1]), contig=True)
        return Tensor(Dispatcher.dispatch(x, ops=UnaryOps.TRANSPOSE), device=x.device, properties=tp)

    # ***** ElementwiseOps *****
    @staticmethod
    def add(x: Tensor, y: Tensor) -> Tensor:
        assert x.device == y.device, f"{x.device} and {y.device} does not match"

        ops.Add().forward(x, y)

    # ***** BinaryOps *****
    @staticmethod
    def matmul(x: Tensor, y: Tensor) -> Tensor:
        # TODO: Check dtype as well
        assert x.shape[-1] == y.shape[0], f"{x.shape} and {y.shape} does not match"
        assert x.device == y.device, f"{x.device} and {y.device} does not match"

        def _backward(): pass

        shape = (x.shape[0], y.shape[1])
        tp = TensorProperties(view=False, offset=0, numel=x.shape[0]*y.shape[1], shape=shape, ndim=len(shape), stride=calculate_stride(shape), contig=True)
        # TODO: How do i ensure data is of same dtype
        return Tensor(Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.MATMUL), device=x.device, dtype=x.dtype, properties=tp)
    
    def __repr__(self) -> str:
        return f"Tensor <dtype: {self.dtype} device: {self.device} view:{self.view} shape: {self.shape}>"

    @property
    def shape(self):
        return self.properties.shape
    
    @property
    def numel(self):
        return self.properties.numel

    @property
    def stride(self):
        return self.properties.stride

    @property
    def view(self):
        return self.properties.view

    @property
    def ndim(self):
        return self.properties.ndim

    @property
    def offset(self):
        return self.properties.offset