from __future__ import annotations

from dataclasses import dataclass

from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, Scalar
from dlgrad.helpers import BinaryOps, BufferOps, CustomOps, UnaryOps, cal_sum_out_shape, calculate_stride, prod_


@dataclass
class BufferMetadata:
    shape: tuple
    numel: int
    stride: tuple
    ndim: int


# TODO: Check all conds such as is shapes are compatible etc here
class Buffer:
    def __init__(self, data: CDataPtr, shape: tuple, device: Device, **kwargs) -> None:
        self.ptr = data # ptr to the array
        self.metadata = BufferMetadata(shape, prod_(shape),
                                       kwargs.get("stride", calculate_stride(shape)),
                                       kwargs.get("ndim", len(shape)))
        self.device = device

    @staticmethod
    def uniform(shape: tuple, device: Device, **kwargs) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BufferOps.UNIFORM, device=device, shape=shape, **kwargs),
            shape=shape, device=device
        )

    @staticmethod
    def full(shape: tuple, fill_value: Scalar, device: Device) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BufferOps.FULL, device=device, shape=shape, fill_value=fill_value),
            shape=shape, device=device
        )

    def sum(self, dim: int = -1) -> Buffer:
        out_shape = cal_sum_out_shape(ndim=self.ndim, dim=dim, inp_shape=self.shape)

        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.SUM, device=self.device, x=self, dim=dim),
            shape=out_shape, device=self.device, ndim=self.ndim if self.ndim == 2 else self.ndim - 1
        )

    def max(self, dim: int = -1) -> tuple[Buffer, Buffer]:
        out_shape = cal_sum_out_shape(ndim=self.ndim, dim=dim, inp_shape=self.shape)

        out, max_with_1s = dispatcher.dispatch(op=UnaryOps.MAX, device=self.device, x=self, dim=dim)

        out_buf = Buffer(data=out, shape=out_shape, device=self.device, ndim=self.ndim if self.ndim == 2 else self.ndim - 1)
        max_with_1s_buf = Buffer(data=max_with_1s, shape=self.shape, device=self.device, ndim=self.ndim if self.ndim == 2 else self.ndim - 1)  # noqa: E501

        return out_buf, max_with_1s_buf

    def matmul(self, other: Buffer) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.MATMUL, device=self.device, x=self, y=other),
            shape=(self.shape[0], other.shape[1]), device=self.device
        )

    # TODO: Check if x is del, then even the transposed is del
    def transpose(self) -> Buffer:
        return Buffer(self.ptr, self.shape[::-1], self.device, stride=self.stride[::-1])

    def exp(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.EXP, device=self.device, x=self),
            shape=self.shape, device=self.device
        )

    def log(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.LOG, device=self.device, x=self),
            shape=self.shape, device=self.device
        )

    def relu(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.RELU, device=self.device, x=self),
            shape=self.shape, device=self.device
        )

    @staticmethod
    def ce_forward(x: Buffer, y: Buffer) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=CustomOps.CE_FORWARD, device=x.device,
                                     x=x, y=y),
            shape=(1, x.shape[0]), device=x.device
        )

    @staticmethod
    def ce_backward(**kwargs) -> None:
        dispatcher.dispatch(op=kwargs.pop("op"), device=kwargs.pop("device"), **kwargs)

    def _binary_op(self, other: Buffer, op: BinaryOps) -> Buffer:
        output_shape = self.shape if self.numel >= other.numel else other.shape

        if op == BinaryOps.SUB and self.numel < other.numel:
            tmp = Buffer(
                data=dispatcher.dispatch(op=BinaryOps.SUB, device=self.device, x=other, y=self),
                shape=other.shape, device=self.device
            )
            return Buffer(
                data=dispatcher.dispatch(op=UnaryOps.NEG, device=self.device, x=tmp),
                shape=tmp.shape, device=tmp.device
            )

        x, y = (self, other) if self.numel >= other.numel else (other, self)
        return Buffer(
            data=dispatcher.dispatch(op=op, device=self.device, x=x, y=y),
            shape=output_shape, device=self.device
        )

    def __add__(self, other: Buffer) -> Buffer:
        return self._binary_op(other, BinaryOps.ADD)

    def __sub__(self, other: Buffer) -> Buffer:
        return self._binary_op(other, BinaryOps.SUB)

    def __mul__(self, other: Buffer) -> Buffer:
        return self._binary_op(other, BinaryOps.MUL)

    def __truediv__(self, other: Buffer) -> Buffer:
        return self._binary_op(other, BinaryOps.DIV)

    def __gt__(self, other: int | float) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.GT, device=self.device, x=self, y=other),
            shape=self.shape, device=self.device
        )

    def __neg__(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.NEG, device=self.device, x=self),
            shape=self.shape, device=self.device
        )

    def __eq__(self, other: Buffer) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.EQT, device=self.device, x=self, y=other),
            shape=self.shape, device=self.device
        )

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
