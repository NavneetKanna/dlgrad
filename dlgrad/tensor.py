from typing import get_args

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dtype import DType, Scalar
from dlgrad.runtime import cpu  # noqa: F401


class OP:
    """
    
    Thanks to tinygrad for the template, this is similar to the Function class.
    """
    def __init__(self) -> None:
        pass
    
    def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

    @classmethod
    def execute(fxn):
        pass


import dlgrad.ops as Op  # noqa: E402


class Tensor:
    def __init__(
            self, data: Scalar | Buffer, device: str | Device | None = None,
            dtype: str | DType | None = None, requires_grad: bool = False
    ) -> None:
        self.device: Device = device if isinstance(device, Device) else Device.from_str(device) if isinstance(device, str) else Device.CPU
        self.dtype: DType = dtype if isinstance(dtype, DType) else DType.from_str(dtype) if isinstance(dtype, str) else DType.FLOAT32
        self.requires_grad: bool = requires_grad
        self._ctx = None

        if isinstance(data, get_args(Scalar)):
            self.data = Op.create_buffer_from_scalar(data, dtype=self.dtype, device=self.device)
