from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from dlgrad.buffer import Buffer
from dlgrad.helpers import Device
from dlgrad.runtime.cpu import CPU

if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class Dispatcher:
    @staticmethod
    def _cpu_dispatch(op, x: Optional[Tensor] = None, y: Optional[Tensor] = None, **kwargs) -> Buffer:
        CPU.interface(op, x, y, **kwargs)

    @staticmethod
    # both the inputs can be None since BufferOps are also dispatched
    def dispatch(ops, x: Tensor = None, y: Tensor = None, **kwargs) -> Buffer:
        device = x.device if x is not None else kwargs["device"]
        if device == Device.CPU:
            return Dispatcher._cpu_dispatch(ops, x, y, **kwargs)
        if device == Device.GPU:
            pass
        else:
            raise ValueError(f"Unsupported device: {device}")
