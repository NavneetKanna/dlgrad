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
    def create_buffer_from_scalar(x: Scalar, device: Device) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BufferOps.CREATE, device=device, x=x),
            shape=tuple(), device=device, ndim=0
        )

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
            shape=out_shape, device=self.device, ndim=self.ndim - 1
        )

    def matmul(self, other: Buffer) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.MATMUL, device=self.device, x=self, y=other),
            shape=(self.shape[0], other.shape[1]), device=self.device
        )

    # TODO: Check if x is del, then even the transposed is del
    def transpose(self) -> Buffer:
        return Buffer(self.ptr, self.shape[::-1], self.device, stride=self.stride[::-1])

    def relu(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.RELU, device=self.device, x=self),
            shape=self.shape, device=self.device
        )

    def __add__(self, other: Buffer) -> Buffer:
        if self.numel >= other.numel:
            return Buffer(
                data=dispatcher.dispatch(op=BinaryOps.ADD, device=self.device, x=self, y=other),
                shape=self.shape, device=self.device
            )
        else:
            return Buffer(
                data=dispatcher.dispatch(op=BinaryOps.ADD, device=self.device, x=other, y=self),
                shape=other.shape, device=self.device
            )

    def __sub__(self, other: Buffer) -> Buffer:
        if self.numel >= other.numel:
            return Buffer(
                data=dispatcher.dispatch(op=BinaryOps.SUB, device=self.device, x=self, y=other),
                shape=self.shape, device=self.device
            )
        else:
            tmp = Buffer(data=dispatcher.dispatch(op=BinaryOps.SUB, device=self.device, x=other, y=self),
                         shape=other.shape, device=self.device)
            tmp = Buffer(data=dispatcher.dispatch(op=BinaryOps.NEG, device=self.device, x=tmp),
                         shape=tmp.shape, device=tmp.device)
            return tmp

    def __mul__(self, other: Buffer) -> Buffer:
        if self.numel >= other.numel:
            return Buffer(
                data=dispatcher.dispatch(op=BinaryOps.MUL, device=self.device, x=self, y=other),
                shape=self.shape, device=self.device
            )
        else:
            return Buffer(
                data=dispatcher.dispatch(op=BinaryOps.MUL, device=self.device, x=other, y=self),
                shape=other.shape, device=self.device
            )

    def __gt__(self, other: int | float) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.GT, device=self.device, x=self, y=other),
            shape=self.shape, device=self.device
        )

    def __neg__(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.NEG, device=self.device, x=self),
            shape=self.shape, device=self.device
        )

    # only for nll loss
    def __getitem__(self, i):  # noqa: ANN001, ANN204
        print("in buffer", i)
        return Buffer(
            data=dispatcher.dispatch(op=CustomOps.INDEX, device=self.device, x=self),
            shape=len(i[0]), device=self.device
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
