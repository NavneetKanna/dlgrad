from __future__ import annotations

import atexit
import ctypes
import math
import warnings
from typing import Optional, Union

import numpy as np

from dlgrad.buffer import Buffer
from dlgrad.dispatch import Dispatcher
from dlgrad.dtype import dtypes
from dlgrad.helpers import (BinaryOps, BufferOps, Device, ShapeError, UnaryOps,
                            calculate_nchw_offset, calculate_stride, set_graph, prod, analyse_list)


# TODO: change c code for max to handle higher dim
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


class Tensor:
    # __slots__ = "grad"

    def __init__(
            self, data: Union[list, int, Buffer], requires_grad = True, device: Optional[Device] = Device.CPU,
            dtype: Optional[dtypes] = dtypes.float32, properties: Optional[TensorProperties] = None,
    ):
        self.requires_grad = requires_grad
        self.device = device
        self.properties = properties
        self._ctx = None
        self.grad = None

        if isinstance(data, Union[int, float]):
            self.data = data
            self._len = 1
            self.dtype = dtype if dtype else dtypes.from_py(data)

        elif isinstance(data, list):
            out_len, out_shape, ndim = analyse_list(data)

            tp = TensorProperties(
                view=False, offset=0, numel=out_len, shape=out_shape,
                ndim=ndim, stride=calculate_stride(out_shape), contig=True, metadata={"created_by": "", "ops": "BufferOps"},
            )
            self.data = Dispatcher.dispatch(ops=BufferOps.CUSTOM, li=data, device=device, func="from_list")
            self.dtype = dtype
            self.device = device
            self.properties=tp

        elif isinstance(data, Buffer):
            self.data = data
            self.dtype = dtype

        # TODO: A better way to write this, a queue ?
        if (not self.properties.view) and isinstance(self.data, Buffer) and self.properties.numel != 1 and not isinstance(data, list):
            atexit.register(self.cleanup)

    def numpy(self) -> np.ndarray:
        if not isinstance(self.data, Buffer):
            pass
        # elif self.numel == 1 and not self.view:
        #     print(self.data.buffer, type(self.data.buffer))
        #     return self.data.buffer.contents
        else:
            sd = ctypes.addressof(self.data.buffer.contents) + self.offset * ctypes.sizeof(ctypes.c_float)
            ptr = (ctypes.c_float * self.numel).from_address(sd)
            data = np.frombuffer(ptr, count=self.numel, dtype=np.float32).reshape(self.shape)
            return data

    def linear(self, weight: Tensor, bias: Tensor) -> Tensor:
        # self@weight.T + bias
        return Tensor.add(Tensor.matmul(self, Tensor.transpose(weight)), bias)

    # TODO: maybe do a check for data before calling free ?
    def cleanup(self):
        Buffer.free(self.data.buffer)

    def __getitem__(self, indices):
        # TODO: slices
        # NOTE: dlgrad is NCHW
        
        # basic indexing 
        if isinstance(indices, int):
            if indices > self.shape[0]:
                raise IndexError(f"index {indices} is out of bounds with {self.shape[0]}")

            offset = calculate_nchw_offset(h=indices, H=self.stride[0])
            tp = TensorProperties(
                view=True, offset=self.offset+offset, numel=prod(self.shape[1:]), shape=self.shape[1:],
                ndim=len(self.shape[1:]), stride=self.stride[1:], contig=True
            )
            return Tensor(Buffer(self.data.buffer), device=self.device, properties=tp)

        if isinstance(indices, Tensor) and self.ndim == 1:
                return Buffer.create_buf_from_idx(self, indices)

    # ***** BufferOps *****
    @staticmethod
    def rand(
        *shape, low=0.0, high=1.0,
        device: Optional[Device] = Device.CPU, dtype: Optional[dtypes] = dtypes.float32,
    ) -> Tensor:
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

        tp = TensorProperties(
            view=False, offset=0, numel=out_len, shape=shape,
            ndim=len(shape), stride=calculate_stride(shape), contig=True, metadata={"created_by": "rand", "ops": "BufferOps"},
        )
        return Tensor(
            Dispatcher.dispatch(ops=BufferOps.UNIFORM, out_len=out_len, low=low, high=high, device=device),
            device=device, dtype=dtype, properties=tp,
        )

    @staticmethod
    def ones(*shape, device: Optional[Device] = Device.CPU, dtype: Optional[dtypes] = dtypes.float32) -> Tensor:
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
            view=False, offset=0, numel=out_len, shape=shape,
            ndim=len(shape), stride=calculate_stride(shape), contig=True, metadata={"created_by": "ones", "ops": "BufferOps"},
        )
        return Tensor(
            Dispatcher.dispatch(ops=BufferOps.ONES, out_len=out_len, device=device),
            device=device, dtype=dtype, properties=tp
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
            view=False, offset=0, numel=out_len, shape=shape,
            ndim=len(shape), stride=calculate_stride(shape), contig=True, metadata={"created_by": "kaiming_uniform", "ops": "BufferOps"},
        )
        return Tensor(
            Dispatcher.dispatch(ops=BufferOps.UNIFORM, out_len=out_len, low=-bound, high=bound, device=device),
            device=device, dtype=dtype, properties=tp
        )

    # ***** UnaryOps ****
    # TODO: What to do if i want to call x.transpose() ?
    @staticmethod
    def transpose(x: Tensor):
        tp = TensorProperties(
            view=False, offset=0, numel=x.numel, shape=x.shape[::-1],
            ndim=len(x.shape[::-1]), stride=calculate_stride(x.shape[::-1]), contig=True
        )
        return Tensor(
            Dispatcher.dispatch(ops=UnaryOps.TRANSPOSE, x=x),
            device=x.device, properties=tp,
        )

    def sum(self, axis=None, keepdim=False):
        from dlgrad.ops import Sum

        return Sum().forward(self, axis, keepdim)

    def max(self):
        from dlgrad.ops import Max

        return Max().forward(self)

    def exp(self):
        from dlgrad.ops import Exp

        return Exp().forward(self)

    def log(self):
        from dlgrad.ops import Log

        return Log().forward(self)

    @staticmethod
    def relu(x: Tensor) -> Tensor:
        from dlgrad.ops import Relu

        return Relu().forward(x)

    @staticmethod
    def softmax(x: Tensor, axis=1):
        # https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
        t = x - x.max()
        u = t.exp()
        return u / u.sum(axis, keepdim=True)

    @staticmethod
    def log_softmax(x: Tensor, axis=1):
        t = x - x.max()
        u = t.exp()
        return t - u.sum(axis, keepdim=True).log()

    # ***** BinaryOps *****
    @staticmethod
    def add(x: Tensor, y: Tensor) -> Tensor:
        from dlgrad.ops import Add

        return Add().forward(x, y)

    @staticmethod
    def div(x: Tensor, y: Tensor) -> Tensor:
        from dlgrad.ops import Div

        return Div().forward(x, y)

    @staticmethod
    def sub(x: Tensor, y: Tensor) -> Tensor:
        from dlgrad.ops import Sub

        return Sub().forward(x, y)

    @staticmethod
    def matmul(x: Tensor, y: Tensor) -> Tensor:
        # TODO: Check dtype as well
        assert x.shape[-1] == y.shape[0], f"{x.shape} and {y.shape} does not match"
        assert x.device == y.device, f"{x.device} and {y.device} does not match"

        shape = (x.shape[0], y.shape[1])
        tp = TensorProperties(
            view=False, offset=0, numel=x.shape[0] * y.shape[1], shape=shape,
            ndim=len(shape), stride=calculate_stride(shape), contig=True
        )
        # TODO: How do i ensure data is of same dtype
        return Tensor(
            Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.MATMUL),
            device=x.device, dtype=x.dtype, properties=tp
        )

    # ***** Loss functions *****
    @staticmethod
    def crossentropy_loss(logits: Tensor, targets: Tensor):
        # NLL(log(softmax(logits)), targets)
        Tensor.log_softmax(logits)

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

    def __sub__(self, other):
        return Tensor.sub(self, other)

    def __truediv__(self, other):
        return Tensor.div(self, other)

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
