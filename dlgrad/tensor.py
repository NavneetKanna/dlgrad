from __future__ import annotations

import atexit
import ctypes
import math
import warnings
from typing import Optional, Union

import numpy as np

from dlgrad.buffer import Buffer
from dlgrad.dispatch import Dispatcher

# import platform
from dlgrad.dtype import dtypes
from dlgrad.helpers import (
    BinaryOps,
    BufferOps,
    Device,
    ShapeError,
    UnaryOps,
    calculate_nchw_offset,
    calculate_stride,
    set_graph,
)


class TensorProperties:
    def __init__(self, **kwargs) -> None:
        self.view: bool = kwargs["view"]
        self.offset: int = kwargs["offset"]
        self.numel: int = kwargs["numel"]
        self.shape: tuple = kwargs["shape"]
        self.ndim: int = kwargs["ndim"]
        self.stride: tuple = kwargs["stride"]
        self.contig: bool = kwargs["contig"]

        self.set_metadata(
            **kwargs.get("metadata", {"created_by": None, "ops": None, "node_id": None})
        )

    def set_metadata(self, created_by=None, ops=None, node_id=None):
        self.metadata = {"created_by": created_by, "ops": ops, "node_id": node_id}


# TODO: Maybe we can load all ctypes files once in the beginning, so that it does not take time to load ?
# TODO: Does it work on tensors with dim > 2 ?
class Tensor:
    # __slots__ = "grad"

    def __init__(
        self,
        data: Union[list, int, Buffer],
        requires_grad=True,
        device: Device = Device.CPU,
        dtype: Optional[dtypes] = None,
        properties: TensorProperties = None,
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

        # TODO: A better way to write this, a queue ?
        # if (not properties.view) and isinstance(data, Buffer) and properties.numel != 1:
        #     atexit.register(self.cleanup)

    def numpy(self):
        if not isinstance(self.data, Buffer) or self.numel == 1:
            data = np.array(self.data.buffer.contents)
            print(data)
        else:
            sd = ctypes.addressof(
                self.data.buffer.contents
            ) + self.offset * ctypes.sizeof(ctypes.c_float)
            ptr = (ctypes.c_float * self.numel).from_address(sd)
            data = np.frombuffer(ptr, count=self.numel, dtype=np.float32).reshape(
                self.shape
            )
            print(data)

    def linear(self, weight: Tensor, bias: Tensor) -> Tensor:
        # self*weight.T + bias
        return Tensor.add(Tensor.matmul(self, Tensor.transpose(weight)), bias)

    # TODO: maybe do a check for data before calling free ?
    def cleanup(self):
        Buffer.free(self.data.buffer)

    # TODO: its complex, refactor later
    def __getitem__(self, indices):
        # TODO: all int, slices
        # NOTE: dlgrad is NCHW

        if isinstance(indices, int):
            if indices > self.shape[0]:
                raise IndexError(f"index {indices} > {self.shape[0]} of {self.shape}")

            offset = calculate_nchw_offset(h=indices, H=self.stride[0])
            size = self.stride[0]
            tp = TensorProperties(
                view=True,
                offset=offset,
                numel=size,
                shape=(size,),
                ndim=len(size),
                stride=calculate_stride(size),
                contig=True,
            )
            return Tensor(Buffer(self.data.buffer), device=self.device, properties=tp)

        if self.ndim == 2:
            if isinstance(indices, tuple):
                h, w = indices
                if h > self.shape[0]:
                    raise IndexError(
                        f"index {indices} > {self.shape[0]} of {self.shape}"
                    )
                if w > self.shape[1]:
                    raise IndexError(
                        f"index {indices} > {self.shape[1]} of {self.shape}"
                    )

                offset = calculate_nchw_offset(h=h, w=w, H=self.stride[0])
                size = 1
                tp = TensorProperties(
                    view=True,
                    offset=offset,
                    numel=size,
                    shape=(size,),
                    ndim=len(size),
                    stride=calculate_stride(size),
                    contig=True,
                )
                return Tensor(
                    Buffer(self.data.buffer), device=self.device, properties=tp
                )

        if self.ndim == 3:
            if isinstance(indices, tuple):
                length = len(indices)
                if length == 3:
                    c, h, w = indices
                    if c > self.shape[0]:
                        raise IndexError(
                            f"index {indices} > {self.shape[0]} of {self.shape}"
                        )
                    if h > self.shape[1]:
                        raise IndexError(
                            f"index {indices} > {self.shape[1]} of {self.shape}"
                        )
                    if w > self.shape[2]:
                        raise IndexError(
                            f"index {indices} > {self.shape[2]} of {self.shape}"
                        )

                    offset = calculate_nchw_offset(
                        c=c, h=h, w=w, C=self.stride[0], H=self.stride[1]
                    )
                    size = 1
                    tp = TensorProperties(
                        view=True,
                        offset=offset,
                        numel=size,
                        shape=(size,),
                        ndim=len(size),
                        stride=calculate_stride(size),
                        contig=True,
                    )
                    return Tensor(
                        Buffer(self.data.buffer), device=self.device, properties=tp
                    )
                if length == 2:
                    c, h = indices
                    if c > self.shape[0]:
                        raise IndexError(
                            f"index {indices} > {self.shape[0]} of {self.shape}"
                        )
                    if h > self.shape[1]:
                        raise IndexError(
                            f"index {indices} > {self.shape[1]} of {self.shape} "
                        )

                    offset = calculate_nchw_offset(
                        c=c, h=h, C=self.stride[0], H=self.stride[1]
                    )
                    size = self.stride[1]
                    tp = TensorProperties(
                        view=True,
                        offset=offset,
                        numel=size,
                        shape=(size,),
                        ndim=len(size),
                        stride=calculate_stride(size),
                        contig=True,
                    )
                    return Tensor(
                        Buffer(self.data.buffer), device=self.device, properties=tp
                    )

        if self.ndim == 4:
            if type(indices) == tuple:
                length = len(indices)
                if length == 4:
                    n, c, h, w = indices
                    if n > self.shape[0]:
                        raise IndexError(
                            f"index {indices} > {self.shape[0]} of {self.shape}"
                        )
                    if c > self.shape[1]:
                        raise IndexError(
                            f"index {indices} > {self.shape[1]} of {self.shape}"
                        )
                    if h > self.shape[2]:
                        raise IndexError(
                            f"index {indices} > {self.shape[2]} of {self.shape}"
                        )
                    if w > self.shape[3]:
                        raise IndexError(
                            f"index {indices} > {self.shape[3]} of {self.shape}"
                        )

                    offset = calculate_nchw_offset(
                        n=n,
                        c=c,
                        h=h,
                        w=w,
                        N=self.stride[0],
                        C=self.stride[1],
                        H=self.stride[2],
                    )
                    size = 1
                    tp = TensorProperties(
                        view=True,
                        offset=offset,
                        numel=size,
                        shape=(size,),
                        ndim=len(size),
                        stride=calculate_stride(size),
                        contig=True,
                    )
                    return Tensor(
                        Buffer(self.data.buffer), device=self.device, properties=tp
                    )
                if length == 3:
                    n, c, h = indices
                    if n > self.shape[0]:
                        raise IndexError(
                            f"index {indices} > {self.shape[0]} of {self.shape}"
                        )
                    if w > self.shape[1]:
                        raise IndexError(
                            f"index {indices} > {self.shape[1]} of {self.shape}"
                        )
                    if h > self.shape[2]:
                        raise IndexError(
                            f"index {indices} > {self.shape[2]} of {self.shape}"
                        )

                    offset = calculate_nchw_offset(
                        n=n,
                        c=c,
                        h=h,
                        N=self.stride[0],
                        C=self.stride[1],
                        H=self.stride[2],
                    )
                    size = self.stride[2]
                    tp = TensorProperties(
                        view=True,
                        offset=offset,
                        numel=size,
                        shape=(size,),
                        ndim=len(size),
                        stride=calculate_stride(size),
                        contig=True,
                    )
                    return Tensor(
                        Buffer(self.data.buffer), device=self.device, properties=tp
                    )
                if length == 2:
                    n, c = indices
                    if n > self.shape[0]:
                        raise IndexError(
                            f"index {indices} > {self.shape[0]} of {self.shape}"
                        )
                    if c > self.shape[1]:
                        raise IndexError(
                            f"index {indices} > {self.shape[1]} of {self.shape}"
                        )

                    offset = calculate_nchw_offset(
                        n=n, c=c, N=self.stride[0], C=self.stride[1]
                    )
                    size = self.stride[1]
                    tp = TensorProperties(
                        view=True,
                        offset=offset,
                        numel=size,
                        shape=(size,),
                        ndim=len(size),
                        stride=calculate_stride(size),
                        contig=True,
                    )
                    return Tensor(
                        Buffer(self.data.buffer), device=self.device, properties=tp
                    )

    # ***** BufferOps *****
    # BufferOps as of now uses only cpu to generate data
    @staticmethod
    def rand(
        *shape,
        low=0.0,
        high=1.0,
        device: Device = Device.CPU,
        dtype: Optional[dtypes] = dtypes.float32,
    ) -> Tensor:
        if device != Device.CPU:
            warnings.warn("Currently BufferOps are only created on CPU.")

        if dtype != dtypes.float32:
            warnings.warn(
                "Currently dlgrad only supports float32, but more dtypes coming in future. Creating data with dtype=float32."
            )

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

        tp = TensorProperties(
            view=False,
            offset=0,
            numel=out_len,
            shape=shape,
            ndim=len(shape),
            stride=calculate_stride(shape),
            contig=True,
            metadata={"created_by": "rand", "ops": "BufferOps"},
        )
        return Tensor(
            Dispatcher.dispatch(
                ops=BufferOps.UNIFORM,
                out_len=out_len,
                low=low,
                high=high,
                device=device,
            ),
            device=device,
            dtype=dtype,
            properties=tp,
        )

    @staticmethod
    def ones(
        *shape, device: Device = Device.CPU, dtype: Optional[dtypes] = dtypes.float32
    ) -> Tensor:
        if device != Device.CPU:
            warnings.warn("Currently BufferOps are only created on CPU.")

        if dtype != dtypes.float32:
            warnings.warn(
                "Currently dlgrad only supports float32, but more dtypes coming in future. Creating data with dtype=float32."
            )

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

        tp = TensorProperties(
            view=False,
            offset=0,
            numel=out_len,
            shape=shape,
            ndim=len(shape),
            stride=calculate_stride(shape),
            contig=True,
            metadata={"created_by": "ones", "ops": "BufferOps"},
        )
        return Tensor(
            Dispatcher.dispatch(ops=BufferOps.ONES, out_len=out_len, device=device),
            device=device,
            dtype=dtype,
            properties=tp,
        )

    @staticmethod
    def kaiming_uniform(*shape, device=Device.CPU, dtype=dtypes.float32):
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

        tp = TensorProperties(
            view=False,
            offset=0,
            numel=out_len,
            shape=shape,
            ndim=len(shape),
            stride=calculate_stride(shape),
            contig=True,
            metadata={"created_by": "kaiming_uniform", "ops": "BufferOps"},
        )
        return Tensor(
            Dispatcher.dispatch(
                ops=BufferOps.UNIFORM, out_len=out_len, low=-bound, high=bound
            ),
            device=device,
            dtype=dtype,
            properties=tp,
        )

    # ***** UnaryOps ****
    # TODO: What to do if i want to call x.transpose() ?
    @staticmethod
    def transpose(x: Tensor):
        tp = TensorProperties(
            view=False,
            offset=0,
            numel=x.numel,
            shape=x.shape[::-1],
            ndim=len(x.shape[::-1]),
            stride=calculate_stride(x.shape[::-1]),
            contig=True,
        )
        return Tensor(
            Dispatcher.dispatch(ops=UnaryOps.TRANSPOSE, x=x),
            device=x.device,
            properties=tp,
        )

    def sum(self):
        from dlgrad.ops import Sum

        return Sum().forward(self)

    # ***** BinaryOps *****
    @staticmethod
    def add(x: Tensor, y: Tensor) -> Tensor:
        from dlgrad.ops import Add, Broadcast

        # TODO: Check in broadcast ?
        if not x.shape == y.shape:
            out_shape = Broadcast().forward(x, y)
        else:
            out_shape = x.shape

        return Add().forward(x, y, out_shape)

    @staticmethod
    def matmul(x: Tensor, y: Tensor) -> Tensor:
        # TODO: Check dtype as well
        assert x.shape[-1] == y.shape[0], f"{x.shape} and {y.shape} does not match"
        assert x.device == y.device, f"{x.device} and {y.device} does not match"

        def _backward():
            pass

        shape = (x.shape[0], y.shape[1])
        tp = TensorProperties(
            view=False,
            offset=0,
            numel=x.shape[0] * y.shape[1],
            shape=shape,
            ndim=len(shape),
            stride=calculate_stride(shape),
            contig=True,
        )
        # TODO: How do i ensure data is of same dtype
        return Tensor(
            Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.MATMUL),
            device=x.device,
            dtype=x.dtype,
            properties=tp,
        )

    def backward(self):
        set_graph(0)

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v._ctx is not None:
                    for i in v._ctx.parents:
                        build_topo(i)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            # Since input nodes are there as well
            if node._ctx is None:
                continue
            node._ctx.backward(node.grad)

    def __repr__(self) -> str:
        return f"Tensor <dtype: {self.dtype} device: {self.device} view:{self.view} shape: {self.shape}>"

    def __add__(self, other):
        return Tensor.add(self, other)

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
