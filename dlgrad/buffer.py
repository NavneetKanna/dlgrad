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
    cal_sum_out_shape,
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
                                       dtype=dtype, device=device)

    @staticmethod
    def from_scalar(val: Scalar) -> Buffer:
        float_arr = ffi.new("float[]", 1)
        float_arr[0] = val
        return Buffer(data=float_arr, shape=(1, 1), device=Device.CPU, dtype=DType.FLOAT32)

    @staticmethod
    def uniform(shape: tuple, device: Device, dtype: DType | str, **kwargs) -> Buffer:
        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)

        if dtype is not DType.FLOAT32:
            raise NotImplementedError("dlgrad only supports float32")
        if not isinstance(shape, tuple):
            raise ValueError("shape must be a tuple")

        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple")
        if len(shape) == 1:
            shape = (1,) + shape

        return Buffer(
            data=dispatcher.dispatch(op=BufferOps.UNIFORM, device=device, shape=shape, **kwargs),
            shape=shape, device=device, dtype=dtype
        )

    @staticmethod
    def full(shape: tuple, fill_value: Scalar, device: Device, dtype: DType) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BufferOps.FULL, device=device, shape=shape, fill_value=fill_value),
            shape=shape, device=device, dtype=dtype
        )

    def sum(self, dim: int = -1) -> Buffer:
        out_shape = cal_sum_out_shape(ndim=self.ndim, dim=dim, inp_shape=self.shape)

        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.SUM, device=self.device, x=self, dim=dim),
            shape=out_shape, device=self.device,
            ndim=self.ndim if self.ndim == 2 else self.ndim - 1, dtype=self.dtype
        )

    # NOTE: keepdim is true by default
    def max(self, dim: int = -1) -> tuple[Buffer, Buffer]:
        out_shape = cal_sum_out_shape(ndim=self.ndim, dim=dim, inp_shape=self.shape)

        out, max_with_1s = dispatcher.dispatch(op=UnaryOps.MAX, device=self.device, x=self, dim=dim)

        out_buf = Buffer(data=out, shape=out_shape, device=self.device, ndim=self.ndim, dtype=self.dtype)  # noqa: E501
        max_with_1s_buf = Buffer(data=max_with_1s, shape=self.shape, device=self.device, ndim=self.ndim, dtype=self.dtype)  # noqa: E501

        return out_buf, max_with_1s_buf

    def matmul(self, other: Buffer) -> Buffer:
        if (self.shape[-1] != other.shape[0] and self.ndim != 2 and other.ndim != 2):
            raise ValueError("Either the Tensors shape dont match or is not 2D")

        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.MATMUL, device=self.device, x=self, y=other),
            shape=(self.shape[0], other.shape[1]), device=self.device, dtype=self.dtype
        )

    def transpose(self) -> Buffer:
        assert self.ndim == 2, "Only 2D Tensors can be transposed"

        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.TRANSPOSE, device=self.device, x=self),
            shape=self.shape[::-1], device=self.device, dtype=self.dtype
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

    def log(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.LOG, device=self.device, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def relu(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.RELU, device=self.device, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    @staticmethod
    def ce_forward(x: Buffer, y: Buffer) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=CustomOps.CE_FORWARD, device=x.device,
                                     x=x, y=y),
            shape=(1, x.shape[0]), device=x.device, dtype=x.dtype
        )

    @staticmethod
    def ce_backward(**kwargs) -> None:
        dispatcher.dispatch(op=kwargs.pop("op"), device=kwargs.pop("device"), **kwargs)

    def _binary_op(self, other: Buffer | Scalar, op: BinaryOps) -> Buffer:
        if isinstance(other, Scalar):
            return Buffer(
                data=dispatcher.dispatch(op=op, device=self.device, x=self, y=other),
                shape=self.shape, device=self.device, dtype=self.dtype
            )

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
        return Buffer(
            data=dispatcher.dispatch(op=op, device=self.device, x=x, y=y),
            shape=output_shape, device=self.device, dtype=self.dtype
        )

    def __add__(self, other: Buffer | Scalar) -> Buffer:
        return self._binary_op(other, BinaryOps.ADD)

    def __sub__(self, other: Buffer | Scalar) -> Buffer:
        return self._binary_op(other, BinaryOps.SUB)

    def __mul__(self, other: Buffer | Scalar) -> Buffer:
        return self._binary_op(other, BinaryOps.MUL)

    def __truediv__(self, other: Buffer | Scalar) -> Buffer:
        return self._binary_op(other**-1, BinaryOps.MUL)

    def __pow__(self, val: Scalar) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.POW, device=self.device, x=self, val=val),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __gt__(self, other: Scalar) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.GT, device=self.device, x=self, y=other),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __neg__(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.NEG, device=self.device, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __eq__(self, other: Buffer) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.EQT, device=self.device, x=self, y=other),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __matmul__(self, other: Buffer) -> Buffer:
        return self.matmul(other)

    @property
    def T(self) -> Buffer:  # noqa: N802
        return self.transpose()

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
