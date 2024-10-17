from typing import get_args

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dtype import DType, Scalar


class Tensor:
    def __init__(
            self, data: Scalar | list | Buffer, device: str | Device | None = None,
            dtype: str | DType | None = None, requires_grad: bool = False
    ) -> None:
        self.device: Device = device if isinstance(device, Device) else Device.from_str(device) if isinstance(device, str) else Device.CPU
        self.dtype: DType = dtype if isinstance(dtype, DType) else DType.from_str(dtype) if isinstance(dtype, str) else DType.FLOAT32
        self.requires_grad = requires_grad

        if isinstance(data, get_args(Scalar)):
            pass

