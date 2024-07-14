from __future__ import annotations
from dlgrad.runtime.cpu import CPU
from dlgrad.buffer import Buffer
from dlgrad.helpers import BinaryOps, UnaryOps, Device
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class Dispatcher:
    @staticmethod
    def _cpu_dispatch(x: Tensor, ops, y: Tensor = None, **kwargs) -> Buffer:
        if ops == BinaryOps.ADD:
            if x.shape[0] == y.shape[0]:
                return CPU.add_axis0(x, y, x.dtype)
            else:
                return CPU._add_axis1(x, y, x.dtype)
        
        if ops == BinaryOps.MATMUL:
            return CPU.matmul(x, y, x.dtype)  
        
        if ops == UnaryOps.TRANSPOSE:
            return CPU.transpose(x, x.dtype)

        if ops == UnaryOps.SUM:
            # To be explicit
            if kwargs["func"] == CPU.sum_axis0:
                return CPU.sum_axis0(x, x.dtype)
            elif kwargs["func"] == CPU._sum_axis1:
                return CPU._sum_axis1(x, x.dtype)
            elif kwargs["func"] == CPU.sum:
                return CPU.sum(x, x.dtype)
            
    @staticmethod
    def dispatch(x: Tensor, ops, y: Tensor = None, **kwargs) -> Buffer:
        device = x.device
        if device == Device.CPU:
            return Dispatcher._cpu_dispatch(x, ops, y, **kwargs)

        elif device == Device.GPU:
            pass