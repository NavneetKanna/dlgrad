from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from dlgrad.buffer import Buffer
from dlgrad.helpers import BinaryOps, BufferOps, Device, UnaryOps
from dlgrad.runtime.cpu import CPU

if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class Dispatcher:
    @staticmethod
    def _cpu_dispatch(
        op, x: Optional[Tensor] = None, y: Optional[Tensor] = None, **kwargs
    ) -> Buffer:
        axis = kwargs.get("axis", None)

        if isinstance(op, BinaryOps):
            if op == BinaryOps.ADD:
                if axis == 0:
                    return CPU.add_axis0(x, y, x.dtype)
                elif axis == 1:
                    return CPU.add_axis1(x, y, x.dtype)
                return CPU.add(x, y, x.dtype)
            if op == BinaryOps.MATMUL:
                return CPU.matmul(x, y, x.dtype)

        elif isinstance(op, UnaryOps):
            if op == UnaryOps.SUM:
                if axis == 0:
                    return CPU.sum_axis0(x, x.dtype)
                elif axis == 1:
                    return CPU.sum_axis1(x, x.dtype)
                return CPU.sum(x, x.dtype)
            if op == UnaryOps.TRANSPOSE:
                return CPU.transpose(x, x.dtype)
            if op == UnaryOps.MAX:
                return CPU.relu(x)

        elif isinstance(op, BufferOps):
            if op == BufferOps.UNIFORM:
                return CPU.uniform(kwargs["out_len"], kwargs["low"], kwargs["high"])
            if op == BufferOps.ONES:
                return CPU.ones(kwargs["out_len"])

        raise ValueError(f"Unsupported operation: {op}")

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
