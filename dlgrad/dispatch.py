from __future__ import annotations
from dlgrad.runtime.cpu import CPU
from dlgrad.buffer import Buffer
from dlgrad.helpers import BinaryOps, UnaryOps, Device
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class Dispatcher:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _cpu_dispatch(x: Tensor, y: Tensor, ops: str) -> Buffer:
        if ops == BinaryOps.ADD:
            return CPU.add(x, y, x._dtype) 
        
        if ops == BinaryOps.MATMUL:
            return CPU.matmul(x, y, x._dtype)  
        
        if ops == UnaryOps.TRANSPOSE:
            return CPU.transpose(x, x._dtype)

    @staticmethod
    def dispatch(x: Tensor, ops: str, y: Tensor = None) -> Buffer:
        device = x._device
        if device == Device.CPU:
            return Dispatcher._cpu_dispatch(x, y, ops)

        elif device == Device.GPU:
            pass