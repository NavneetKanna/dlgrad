from dataclasses import dataclass

from dlgrad.device import Device
from dlgrad.helpers import calculate_stride, prod_


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