from dataclasses import dataclass

from dlgrad.device import Device
from dlgrad.helpers import calculate_stride, prod_

from dlgrad.dispatch import dispatcher
from dlgrad.helpers import (BinaryOps, BufferOps, UnaryOps)

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

    def neg(self):
        return Buffer(dispatcher.dispatch(op=BinaryOps.NEG, device=self.device, x=self), self.shape, self.device)

    def sum(self):
        return Buffer(dispatcher.dispatch(op=UnaryOps.SUM, device=self.device, x=self), tuple(), self.device, ndim=1)
    
    def matmul(self, other):
        return Buffer(dispatcher.dispatch(op=BinaryOps.MATMUL, device=self.device, x=self, y=other), (self.shape[0], other.shape[1]), self.device)
    
    # TODO: Check if x is del, then even the transposed is del
    def transopose(self):
        return Buffer(self.ptr, self.shape[::-1], self.device, stride=self.stride[::-1])
    
    def __add__(self, other):
        return Buffer(dispatcher.dispatch(op=BinaryOps.ADD, device=self.device, x=self, y=other), self.shape, self.device)

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