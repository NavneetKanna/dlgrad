from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Callable

from dlgrad.buffer import Buffer
from dlgrad.helpers import BinaryOps, BufferOps, Device, UnaryOps
from dlgrad.runtime.cpu import CPU

if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class Dispatcher:

    @staticmethod
    def _cpu_dispatch(ops, x: Optional[Tensor] = None, y: Optional[Tensor] = None, **kwargs) -> Buffer:
        # Binary Ops
        axis = kwargs.get("axis", None)
        if ops == BinaryOps.ADD:
            if axis == 0:
                return CPU.add_axis0(x, y, x.dtype)
            elif axis == 1:
                return CPU._add_axis1(x, y, x.dtype)
            else:
                return CPU.add(x, y, x.dtype)
        elif ops == BinaryOps.MATMUL:
            return CPU.matmul(x, y, x.dtype)  
        
        # Unary Ops
        if ops == UnaryOps.TRANSPOSE:
            return CPU.transpose(x, x.dtype)
        elif ops == UnaryOps.SUM:
            sum_func: Callable = kwargs["func"]
            return sum_func(x, x.dtype)

        if ops == BufferOps.UNIFORM:
            return CPU.uniform(kwargs["out_len"], kwargs["low"], kwargs["high"])

        if ops == BufferOps.ONES:
            return CPU.ones(kwargs["out_len"])

    @staticmethod
    def dispatch(ops, x: Tensor = None, y: Tensor = None, **kwargs) -> Buffer:
        device = x.device if x is not None else kwargs['device']
        if device == Device.CPU:
            return Dispatcher._cpu_dispatch(ops, x, y, **kwargs)

        elif device == Device.GPU:
            pass