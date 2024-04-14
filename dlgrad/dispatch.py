from __future__ import annotations
from dlgrad.runtime.cpu import CPU
from dlgrad.buffer import Buffer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class Dispatcher:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _cpu_dispatch(x: Tensor, y: Tensor, ops: str) -> Buffer:
        if ops == 'add':
            return CPU.add(x, y, x._dtype) 

    @staticmethod
    def dispatch(x: Tensor, y: Tensor, ops: str) -> Buffer:
        device = x._device

        if device == 'cpu':
            return Dispatcher._cpu_dispatch(x, y, ops)

        elif device == 'gpu':
            pass