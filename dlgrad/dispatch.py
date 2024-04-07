from __future__ import annotations
from dlgrad.runtime.cpu import CPU
from dlgrad.buffer import Buffer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class Dispatcher:
    def __init__(self) -> None:
        pass

    def dispatch(x: Tensor, y: Tensor) -> Buffer:
        dtype = x._dtype 
        device = x._device

        if device == 'cpu':
            return CPU.add(x, y, dtype) 

        elif device == 'gpu':
            pass