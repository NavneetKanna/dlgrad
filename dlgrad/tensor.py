"""
1) how to detect a left tensor in computational graph ?
2) AssertionError: backward can only be called for scalar tensors
3) permutation
4) kernel dispatch
5) what are all these __slots__ __fields__ ?
6) scheduling
7) tile size in matmul is l1 cache
8) cache aware matmul
9) hardware command queue
10) multiply-accumulate
11) kernel fusion
"""

from __future__ import annotations
from typing import Union, Optional
from dlgrad.helpers import ShapeError, IndexError, calculate_stride, calculate_nchw_offset, BinaryOps, Device
import ctypes
import atexit
import numpy as np
# import platform
from dlgrad.dtype import dtypes
from dlgrad.buffer import Buffer
import warnings
from dlgrad.dispatch import Dispatcher


class Tensor:
    """
    as of now i will go with NCHW, but later if required i will change to channel last
    """
    # __slots__ = "grad"

    # TODO: is it a good programming practice to have many args ?
    def __init__(
            self,
            data: Union[list, int, Buffer],
            requires_grad = True,
            device: Device = Device.CPU,
            view: bool = False,
            dtype: Optional[dtypes] = None,
            _offset: int = 0,
            _len: int = 0,
            _shape: tuple = ()
        ):
        """
        _len = Number of elements, for ex, (4, 2) -> 8, (2, 3, 4) -> 24
        """

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
            self._offset = _offset
            self._len = _len
            self._shape = _shape
            self._ndim = len(self._shape)
            self._strides = calculate_stride(_shape)
            self._view = view
            self._contig = True
            self._dtype = dtype

        if not view and isinstance(data, Buffer): 
            atexit.register(self.cleanup)

    def numpy(self):
        sd = ctypes.addressof(self._data.contents) + self._offset * ctypes.sizeof(ctypes.c_float)
        ptr = (ctypes.c_float * self._len).from_address(sd)
        data = np.frombuffer(ptr, count=self._len, dtype=np.float32).reshape(self._shape)
        print(data)

    def __repr__(self) -> str:
        return f"Tensor <contig: {self._contig} view:{self._view} device: {self._device}>"

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
            return Tensor(Buffer(self._data), device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))

        if self._ndim == 2:
            if type(indices) == tuple:
                h, w = indices
                if h > self._shape[0]:
                    raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                if w > self._shape[1]:
                    raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape}")

                offset = calculate_nchw_offset(h=h, w=w, H=self._strides[0])
                size = 1
                return Tensor(Buffer(self._data), device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))

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
                    return Tensor(Buffer(self._data), device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))
                elif length == 2:
                    c, h = indices
                    if c > self._shape[0]:
                        raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                    if h > self._shape[1]:
                        raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape} ")

                    offset = calculate_nchw_offset(c=c, h=h, C=self._strides[0], H=self._strides[1])
                    size = self._strides[1]
                    return Tensor(Buffer(self._data), device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))

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
                    return Tensor(Buffer(self._data), device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))
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
                    return Tensor(Buffer(self._data), device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))
                elif length == 2:
                    n, c  = indices
                    if n > self._shape[0]:
                        raise IndexError(f"index {indices} > {self._shape[0]} of {self._shape}")
                    if c > self._shape[1]:
                        raise IndexError(f"index {indices} > {self._shape[1]} of {self._shape}")

                    offset = calculate_nchw_offset(n=n, c=c, N=self._strides[0], C=self._strides[1])
                    size = self._strides[1]
                    return Tensor(Buffer(self._data), device=self._device, view=True, _offset=offset, _len=size, _shape=(size,))

    # ***** BufferOps *****
    # BufferOps as of now uses only cpu to generate data
    @staticmethod
    def rand(*shape, device: Device = Device.CPU, dtype: Optional[dtypes] = dtypes.float32) -> Tensor:
        if device != Device.CPU:
            warnings.warn("As of now DCOPS (data creation ops) are only created on CPU.")

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
        size = 1
        for i in shape: 
            size *= i
        
        return Tensor(Buffer.create_random_buffer(size), _offset=0, device=device, dtype=dtype, _len=size, _shape=shape)

    # TODO: where is broadcasting used ?
    # TODO: support +
    # TODO: add enum of ops, device
    # ***** BinaryOps *****
    @staticmethod
    def add(x: Tensor, y: Tensor) -> Tensor:
        assert x._shape == y._shape, f"{x._shape} and {y._shape} does not match"
        assert x._device == y._device, f"{x._device} and {y._device} does not match"

        def _backward(): pass

        return Tensor(Dispatcher.dispatch(x, y, ops=BinaryOps.ADD), device=x._device, _len=x._len, _shape=x._shape, view=False)
    
    @staticmethod
    def matmul(x: Tensor, y: Tensor) -> Tensor:
        assert x._shape[::-1] == y._shape, f"{x._shape} and {y._shape} does not match"
        assert x._device == y._device, f"{x._device} and {y._device} does not match"

        def _backward(): pass

        # TODO: How do i ensure data is of same dtype
        return Tensor(Dispatcher.dispatch(x, y, ops=BinaryOps.MATMUL), device=x._device, _len=x._shape[0]*y._shape[1], _shape=(x._shape[0], y._shape[1]), view=False)

"""
clean git history
does .so file work for windows

further reading

https://sidsite.com/posts/autodiff/

https://triton-lang.org/main/_images/grouped_vs_row_major_ordering.png
https://triton-lang.org/main/_images/triton-parallel-matmul.png

Standard ARMv8 SIMD/NEON vector instructions on CPU cores (128 bits wide, issue up to four per cycle on Firestorm)

https://dougallj.github.io/applecpu/firestorm-simd.html

https://arxiv.org/pdf/2001.05585.pdf

https://www.youtube.com/watch?v=wIPdrbZIeKE

https://www.cise.ufl.edu/~sahni/papers/gpuMatrixMultiply.pdf

arxiv metal

Also note that M1 is only ARMv8.4-A compliant

https://godbolt.org

Convolutions can also be cast as matrix multiplications. For example, a classic approach for this is to preprocess the input using im2col, 
do a matrix multiplication, and then apply col2im. This is very inefficient, because the im2col matrix needs to be generated, and takes up a lot 
of extra memory and time. However, this approach can be made efficient using implicit im2col, resulting in an implicit GEMM convolution, which is 
implemented in some of the cuDNN algorithms for NVIDIA GPUs. Also, a 1x1 convolution can be cast as a matrix multiplication even without im2col.
https://datascience.stackexchange.com/questions/12830/how-are-1x1-convolutions-the-same-as-a-fully-connected-layer

JIT 

what is the compute of matmul ?

"""
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
