from __future__ import annotations
from dlgrad.runtime.cpu import CPU
from dlgrad.buffer import Buffer
from dlgrad.helpers import BinaryOps, UnaryOps, Device
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class Dispatcher:
    @staticmethod
    def _cpu_dispatch(x: Tensor, y: Tensor, ops) -> Buffer:
        if ops == BinaryOps.ADD:
            return CPU.add(x, y, x.dtype) 
        
        if ops == BinaryOps.MATMUL:
            return CPU.matmul(x, y, x.dtype)  
        
        if ops == UnaryOps.TRANSPOSE:
            return CPU.transpose(x, x.dtype)

    @staticmethod
    def dispatch(x: Tensor, ops, y: Tensor = None) -> Buffer:
        device = x.device
        if device == Device.CPU:
            return Dispatcher._cpu_dispatch(x, y, ops)

        elif device == Device.GPU:
            pass