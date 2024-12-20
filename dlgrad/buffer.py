from __future__ import annotations

from dataclasses import dataclass

from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import Scalar
from dlgrad.helpers import (BinaryOps, BufferOps, UnaryOps, calculate_stride,
                            prod_)


@dataclass
class BufferMetadata:
    shape: tuple
    numel: int
    stride: tuple
    ndim: int


class Buffer:
    def __init__(self, data, shape: tuple, device: Device, **kwargs) -> None:
        self.ptr = data # ptr to the array
        self.metadata = BufferMetadata(shape, prod_(shape), kwargs.get("stride", calculate_stride(shape)), kwargs.get("ndim", len(shape)))
        self.device = device

    def neg(self) -> Buffer:
        return Buffer(dispatcher.dispatch(op=BinaryOps.NEG, device=self.device, x=self), self.shape, self.device)

    def sum(self, dim: int) -> Buffer:
        out_shape = tuple()
        ndim = 0
        if self.ndim == 3:
            ndim = 2
            if dim == 0:
                out_shape = (self.shape[1], self.shape[2])
            elif dim == 1:
                out_shape = (self.shape[0], self.shape[2])
            elif dim == 2:
                out_shape = (self.shape[0], self.shape[1])
        
        return Buffer(dispatcher.dispatch(op=UnaryOps.SUM, device=self.device, x=self, dim=dim, numel=prod_(out_shape)), out_shape, self.device, ndim=ndim)
    
    def matmul(self, other) -> Buffer:
        return Buffer(dispatcher.dispatch(op=BinaryOps.MATMUL, device=self.device, x=self, y=other), (self.shape[0], other.shape[1]), self.device)
    
    # TODO: Check if x is del, then even the transposed is del
    def transpose(self) -> Buffer:
        return Buffer(self.ptr, self.shape[::-1], self.device, stride=self.stride[::-1])
    
    @staticmethod
    def create_buffer_from_scalar(x: Scalar, device: Device) -> Buffer:
        return Buffer(dispatcher.dispatch(op=BufferOps.CREATE, device=device, x=x), tuple(), device, ndim=0)

    @staticmethod
    def uniform(shape: tuple, device: Device, **kwargs) -> Buffer:
        return Buffer(dispatcher.dispatch(op=BufferOps.UNIFORM, device=device, shape=shape, **kwargs), shape, device)

    @staticmethod
    def full(shape: tuple, fill_value: Scalar, device: Device) -> Buffer:
        return Buffer(dispatcher.dispatch(op=BufferOps.FULL, device=device, shape=shape, fill_value=fill_value), shape, device)

    def __add__(self, other) -> Buffer:
        if self.numel > other.numel or self.numel == other.numel:
            return Buffer(dispatcher.dispatch(op=BinaryOps.ADD, device=self.device, x=self, y=other), self.shape, self.device)
        else:
            return Buffer(dispatcher.dispatch(op=BinaryOps.ADD, device=self.device, x=self, y=other), other.shape, self.device)

    def __sub__(self, other) -> Buffer:
        if self.numel > other.numel or self.numel == other.numel:
            return Buffer(dispatcher.dispatch(op=BinaryOps.SUB, device=self.device, x=self, y=other), self.shape, self.device)
        else:
            return Buffer(dispatcher.dispatch(op=BinaryOps.SUB, device=self.device, x=self, y=other), other.shape, self.device)
    
    @property
    def numel(self):
        return self.metadata.numel

    @property
    def shape(self):
        return self.metadata.shape
    
    @property
    def stride(self):
        return self.metadata.stride
    
    @property
    def ndim(self):
        return self.metadata.ndim