from __future__ import annotations

from dataclasses import dataclass

from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, DType, Scalar
from dlgrad.helpers import (
    BinaryOps,
    BufferOps,
    CustomOps,
    UnaryOps,
    cal_sum_max_out_shape,
    calculate_stride,
    check_broadcast,
    ffi,
    prod_,
)


@dataclass
class BufferMetadata:
    shape: tuple
    numel: int
    stride: tuple
    ndim: int
    dtype: DType
    device: Device
    nbytes: int


class Buffer:
    def __init__(
            self, data: CDataPtr, shape: tuple,
            device: str | Device | None = Device.CPU,
            dtype: str | DType | None = None, **kwargs
        ) -> None:
        self.ptr = data # ptr to the array
        self.metadata = BufferMetadata(shape=shape, numel=prod_(shape),
                                       stride=kwargs.get("stride", calculate_stride(shape)),
                                       ndim=kwargs.get("ndim", len(shape)),
                                       dtype=dtype, device=device, nbytes=prod_(shape)*DType.get_n_bytes(dtype))

    @staticmethod
    def from_scalar(val: Scalar) -> Buffer:
        float_arr = ffi.new("float[]", 1)
        float_arr[0] = val
        return Buffer(data=float_arr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)

    @staticmethod
    def uniform(shape: tuple, device: Device, dtype: DType | str, **kwargs) -> Buffer:
        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)
        if isinstance(device, str):
            device = Device.from_str(device)

        if dtype is not DType.FLOAT32:
            raise NotImplementedError("dlgrad only supports float32")
        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple")

        # All data creation is done on cpu
        return Buffer(
            data=dispatcher.dispatch(op=BufferOps.UNIFORM,
                                     device=Device.CPU if device != Device.CPU else device,
                                     shape=shape, **kwargs),
            shape=shape, device=device, dtype=dtype
        )

    @staticmethod
    def full(shape: tuple, fill_value: Scalar, device: Device, dtype: DType) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BufferOps.FULL, device=Device.CPU, shape=shape, fill_value=fill_value),
            shape=shape, device=device, dtype=dtype
        )

    def unsqueeze(self, dim: int) -> None:
        if dim == -1:
            dim = 0
        assert dim <= len(self.shape), f"Cannot unsqueeze {dim} from {self.shape}"

        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        self.metadata.shape = tuple(new_shape)
        self.metadata.numel = prod_(new_shape)
        self.metadata.stride = calculate_stride(new_shape)
        self.metadata.ndim = len(new_shape)

    def squeeze(self, dim: list[int]) -> None:
        for idx in dim:
            if self.shape[idx] != 1:
                continue

            new_shape = list(self.shape)
            new_shape.pop(idx)
            self.metadata.shape = tuple(new_shape)
            self.metadata.numel = prod_(new_shape)
            self.metadata.stride = calculate_stride(new_shape)
            self.metadata.ndim = len(new_shape)

    def sum(self, dim: int = -1, keepdim: bool = False) -> Buffer:
        out_shape = cal_sum_max_out_shape(ndim=self.ndim, dim=dim, inp_shape=self.shape, keepdim=keepdim)

        if keepdim:
            ndim = self.ndim
        else:
            if dim == -1:
                ndim = 0
            else:
                ndim = self.ndim - 1

        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.SUM, device=self.device, x=self, dim=dim),
            shape=out_shape, device=self.device,
            ndim=ndim, dtype=self.dtype
        )

    def mean(self, dim: int = -1, keepdim: bool = False) -> Buffer:
        out_shape = cal_sum_max_out_shape(ndim=self.ndim, dim=dim, inp_shape=self.shape, keepdim=keepdim)

        if keepdim:
            ndim = self.ndim
        else:
            if dim == -1:
                ndim = 0
            else:
                ndim = self.ndim - 1

        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.MEAN, device=Device.CPU, x=self, dim=dim),
            shape=out_shape, device=self.device,
            ndim=ndim, dtype=self.dtype
        )

    def max(self, dim: int = -1, backward: bool = False, out: Buffer = None, keepdim: bool = False) -> Buffer:
        out_shape = cal_sum_max_out_shape(ndim=self.ndim, dim=dim, inp_shape=self.shape, keepdim=keepdim)

        if not backward:
            out = dispatcher.dispatch(op=UnaryOps.MAX, device=Device.CPU, x=self, dim=dim, backward=backward)
        else:
            out_shape = self.shape
            out = dispatcher.dispatch(op=UnaryOps.MAX, device=Device.CPU, x=self, dim=dim, backward=backward, out=out)

        if keepdim:
            ndim = self.ndim
        else:
            if dim == -1:
                ndim = 0
            else:
                ndim = self.ndim - 1

        out_buf = Buffer(data=out, shape=out_shape, device=self.device, ndim=ndim, dtype=self.dtype)  # noqa: E501

        return out_buf

    def matmul(self, other: Buffer) -> Buffer:
        assert self.ndim == 2 and other.ndim == 2, "dlgrad only supports 2d matrix multiplication"
        if (self.shape[-1] != other.shape[0] and self.ndim != 2 and other.ndim != 2):
            raise ValueError("Either the Tensors shape dont match or is not 2D")

        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.MATMUL, device=self.device, x=self, y=other),
            shape=(self.shape[0], other.shape[1]), device=self.device, dtype=self.dtype
        )

    @staticmethod
    def swap_indices(inp: tuple, tmp: tuple) -> tuple:
        lst = list(inp)
        for i in range(0, len(tmp), 2):
            if i+1 < len(tmp):
                idx1, idx2 = tmp[i], tmp[i+1]
                lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
            return tuple(lst)

    def transpose(self, axes: tuple) -> Buffer:
        # assert self.ndim == 2, "Only 2D Tensors can be transposed"

        return Buffer(
            data=dispatcher.dispatch(
                op=UnaryOps.TRANSPOSE,
                device=Device.CPU,
                x=self,
                out_shape=Buffer.swap_indices(self.shape, axes),
                out_stride=calculate_stride(Buffer.swap_indices(self.shape, axes)),
                axes=axes
            ),
            shape=Buffer.swap_indices(self.shape, axes), device=self.device, dtype=self.dtype
        )

    def exp(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.EXP, device=self.device, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def sqrt(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.SQRT, device=self.device, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def clamp(self, min: int | None = None, max: int | None = None) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.CLAMP, device=Device.CPU, x=self, min=min, max=max),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def log(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.LOG, device=self.device, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def relu(self) -> Buffer:
        return Buffer(
            data=self.where(inp=self, other=0.0).ptr,
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def sigmoid(self) -> Buffer:
        out = self.exp() / (self.exp() + 1.0)
        return Buffer(
            data=out.ptr,
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def leaky_relu(self, neg_slope: Scalar = 0.01) -> Buffer:
        return Buffer(
            data=self.where(inp=self, other=self*neg_slope).ptr,
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def tanh(self) -> Buffer:
        out = (self.exp() - (-self).exp()) / (self.exp() + (-self).exp())
        return Buffer(
            data=out.ptr,
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    @staticmethod
    def ce_forward(x: Buffer, y: Buffer) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=CustomOps.CE_FORWARD, device=Device.CPU,
                                     x=x, y=y),
            shape=(1, x.shape[0]), device=x.device, dtype=x.dtype
        )

    def argmax(self, axis: int) -> Buffer:
        assert self.ndim == 2, "currently dlgrad supports argmax for 2d only"
        if axis==0:
            s = (1, self.shape[1])
        elif axis==1:
            s = (self.shape[0], 1)
        else:
            s = (1, 1)
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.ARGMAX, device=Device.CPU,
                                     x=self, axis=axis),
            shape=s, device=self.device, dtype=self.dtype
        )

    def where(self, inp: Buffer | Scalar, other: Buffer | Scalar) -> Buffer:
        if isinstance(inp, Scalar):
            inp = Buffer.from_scalar(inp)
        if isinstance(other, Scalar):
            other = Buffer.from_scalar(other)
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.WHERE, device=Device.CPU,
                                    x=self, inp=inp, other=other),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    @staticmethod
    def ce_backward(**kwargs) -> None:
        kwargs.pop("device")
        dispatcher.dispatch(op=kwargs.pop("op"), device=Device.CPU, **kwargs)

    def _binary_op(self, other: Buffer, op: BinaryOps) -> Buffer:
        if not check_broadcast(self.shape, other.shape):
            raise ValueError(f"Cannot broadcast {other.shape} to {self.shape}")

        output_shape = self.shape if self.numel >= other.numel else other.shape

        if op == BinaryOps.SUB and self.numel < other.numel:
            tmp = Buffer(
                data=dispatcher.dispatch(op=BinaryOps.SUB, device=self.device, x=other, y=self),
                shape=other.shape, device=self.device, dtype=self.dtype
            )
            return Buffer(
                data=dispatcher.dispatch(op=UnaryOps.NEG, device=self.device, x=tmp),
                shape=tmp.shape, device=tmp.device, dtype=self.dtype
            )

        x, y = (self, other) if self.numel >= other.numel else (other, self)

        # if y.shape is different from x.shape, for example, (2, 3, 4) & (3, 4)
        # unsqueeze y.shape to (1, 3, 4) and squeeze it back to (3, 4)
        tmp = []
        org_yshape = y.shape
        if len(x.shape) != len(y.shape) and y.ndim != 1 and y.ndim != 0:
            shape_diff = len(x.shape) - len(y.shape)
            for i in range(shape_diff):
                tmp.append(i)
                y.unsqueeze(0)

        t = Buffer(
            data=dispatcher.dispatch(op=op, device=self.device, x=x, y=y),
            shape=output_shape, device=self.device, dtype=self.dtype
        )

        if len(x.shape) != len(org_yshape) and y.ndim != 1 and y.ndim != 0:
            y.squeeze(tmp)

        return t

    def __add__(self, other: Buffer | Scalar) -> Buffer:
        if isinstance(other, Scalar):
            other = Buffer(data=Buffer.from_scalar(other).ptr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)
        return self._binary_op(other, BinaryOps.ADD)

    def __sub__(self, other: Buffer | Scalar) -> Buffer:
        print("sub called", self, other)
        if isinstance(other, Scalar):
            other = Buffer(data=Buffer.from_scalar(other).ptr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)
        return self._binary_op(other, BinaryOps.SUB)

    def __mul__(self, other: Buffer | Scalar) -> Buffer:
        if isinstance(other, Scalar):
            other = Buffer(data=Buffer.from_scalar(other).ptr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)
        return self._binary_op(other, BinaryOps.MUL)

    def __truediv__(self, other: Buffer | Scalar) -> Buffer:
        if isinstance(other, Scalar):
            other = Buffer(data=Buffer.from_scalar(other).ptr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)

        if self.numel >= other.numel:
            return self._binary_op(other, BinaryOps.DIV)
        else:
            return self._binary_op(other**-1, BinaryOps.MUL)

    def __rsub__(self: Buffer, other: Scalar) -> Buffer:
        if isinstance(other, Scalar):
            other = Buffer(data=Buffer.from_scalar(other).ptr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)

        return -(self._binary_op(other, BinaryOps.SUB))

    def __pow__(self, val: Scalar) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.POW, device=self.device, x=self, val=val),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __gt__(self, other: Scalar) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.GT, device=Device.CPU, x=self, y=other),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __neg__(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.NEG, device=self.device, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __eq__(self, other: Buffer) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.EQT, device=Device.CPU, x=self, y=other),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __matmul__(self, other: Buffer) -> Buffer:
        return self.matmul(other)

    @property
    def T(self) -> Buffer:  # noqa: N802
        return self.transpose(tuple([i for i in range(self.ndim)][::-1]))

    @property
    def numel(self) -> int:
        return self.metadata.numel

    @property
    def shape(self) -> tuple:
        return self.metadata.shape

    @property
    def stride(self) -> tuple:
        return self.metadata.stride

    @property
    def ndim(self) -> int:
        return self.metadata.ndim

    @property
    def dtype(self) -> int:
        return self.metadata.dtype

    @property
    def device(self) -> int:
        return self.metadata.device

    @property
    def nbytes(self) -> int:
        return self.metadata.nbytes
