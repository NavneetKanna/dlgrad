from dataclasses import dataclass
from dlgrad.helpers import prod_, calculate_stride


@dataclass
class BufferMetadata:
    shape: tuple
    numel: int
    stride: tuple
    ndim: int


class Buffer:
    def __init__(self, data, shape: tuple, **kwargs) -> None:
        self.ptr = data # ptr to the array
        self.metadata = BufferMetadata(shape, prod_(shape), calculate_stride(shape), kwargs.get("ndim", len(shape)))

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