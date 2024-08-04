from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from dlgrad.buffer import Buffer
from dlgrad.helpers import BinaryOps, BufferOps, Device, UnaryOps
from dlgrad.runtime.cpu import CPU

if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class Dispatcher:

    @staticmethod
    def _cpu_dispatch(op, x: Optional[Tensor] = None, y: Optional[Tensor] = None, **kwargs) -> Buffer:
        axis = kwargs.get("axis", None)

        if isinstance(op, BinaryOps):
            if op == BinaryOps.ADD:
                return CPU._add(x, y, x.dtype, axis)
            elif op == BinaryOps.MATMUL:
                return CPU._matmul(x, y, x.dtype)  
        
        elif isinstance(op, UnaryOps):
            if op == UnaryOps.SUM:
                return CPU._sum(x, x.dtype, axis)
            elif op == UnaryOps.TRANSPOSE:
                return CPU._transpose(x, x.dtype)

        elif isinstance(op, BufferOps):
            if op == BufferOps.UNIFORM:
                return CPU._uniform(kwargs["out_len"], kwargs["low"], kwargs["high"])
            elif op == BufferOps.ONES:
                return CPU._ones(kwargs["out_len"])

    @staticmethod
    # both the inputs can be None since BufferOps are also dispatched
    def dispatch(ops, x: Tensor = None, y: Tensor = None, **kwargs) -> Buffer:
        device = x.device if x is not None else kwargs['device']
        if device == Device.CPU:
            return Dispatcher._cpu_dispatch(ops, x, y, **kwargs)

        elif device == Device.GPU:
            pass