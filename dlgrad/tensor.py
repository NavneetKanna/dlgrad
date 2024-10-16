from dlgrad.buffer import Buffer
from dlgrad.dtype import DType
from dlgrad.device import Device


class Tensor:
    def __init__(
            self, data: int | float | list | Buffer, device: str | Device | None,
            dtype: str | DType | None, requires_grad: bool = False) -> None:
        self.device: Device = device if isinstance(device, Device) else Device.from_str(device) if isinstance(device, str) else Device.CPU
        self.dtype: DType = dtype if isinstance(dtype, DType) else DType.from_str(dtype) if isinstance(dtype, str) else DType.FLOAT32
        self.requires_grad = requires_grad

    
a = Tensor()

a.device