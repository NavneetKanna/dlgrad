import _add  # type: ignore
import _full  # type: ignore
import _matmul  # type: ignore
import _neg  # type: ignore
import _sum  # type: ignore
import _uniform  # type: ignore
import _arithmetic # type: ignore
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, DType, Scalar
from dlgrad.helpers import BinaryOps, BufferOps, UnaryOps, prod_


# TODO: Calling ffi.gc() twice one after the other leads to error, find alternative
class CPU:
    """
    Main CPU runtime class which handles the logic of calling the compiled C source files.

    This class uses CFFI (C Foreign Function Interface) to interact with C code.
    """
    ffi = FFI()

    @staticmethod
    @dispatcher.register(BufferOps.CREATE, Device.CPU)
    def create_buffer_from_scalar(x: Scalar) -> CDataPtr:
        return CPU.ffi.new(f"{DType.get_c_dtype((x))} [1]", [x])

    @staticmethod
    @dispatcher.register(BufferOps.UNIFORM, Device.CPU)
    def uniform(shape: tuple, low: float, high: float) -> CDataPtr:
        numel = prod_(shape)
        arr = _uniform.lib.uniform(numel, low, high)

        return CPU.ffi.gc(arr, _uniform.lib.free_uniform)
    
    @staticmethod
    @dispatcher.register(BufferOps.FULL, Device.CPU)
    def full(shape: tuple, fill_value: Scalar) -> CDataPtr:
        arr = _full.lib.full(prod_(shape), fill_value)

        return CPU.ffi.gc(arr, _full.lib.free_full)

    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.CPU)
    def add(x: Buffer, y: Buffer) -> CDataPtr:
        if x.numel > y.numel or y.numel == x.numel:
            if not (y_stride := y.stride): # for scalar
                y_stride = [0]
            if len(x.shape) == 2:
                arr = _arithmetic.lib.op_2d(x.ptr, y.ptr, x.numel, y.numel, x.shape, y.shape, x.stride, y_stride, len(y.shape), 0)
            elif len(x.shape) == 3:
                arr = _arithmetic.lib.op_3d(x.ptr, y.ptr, x.numel, y.numel, x.shape, y.shape, x.stride, y.stride, len(y.shape), 0)
        else:
            if not (y_stride := x.stride): # for scalar
                y_stride = [0]
            if len(y.shape) == 2:
                arr = _arithmetic.lib.op_2d(x.ptr, y.ptr, x.numel, y.numel, y.shape, x.shape, y.stride, y_stride, len(x.shape), 0)
            elif len(y.shape) == 3:
                arr = _arithmetic.lib.op_3d(x.ptr, y.ptr, x.numel, y.numel, y.shape, x.shape, y.stride, y_stride, len(x.shape), 0)

        return CPU.ffi.gc(arr, _arithmetic.lib.free_op)
        
    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.CPU)
    def sub(x: Buffer, y: Buffer) -> CDataPtr:
        if x.numel > y.numel or y.numel == x.numel:
            if not (y_stride := y.stride): # for scalar
                y_stride = [0]
            if len(x.shape) == 2:
                arr = _arithmetic.lib.op_2d(x.ptr, y.ptr, x.numel, y.numel, x.shape, y.shape, x.stride, y_stride, len(y.shape), 1)
            elif len(x.shape) == 3:
                arr = _arithmetic.lib.op_3d(x.ptr, y.ptr, x.numel, y.numel, x.shape, y.shape, x.stride, y_stride, len(y.shape), 1)
        else:
            if not (y_stride := x.stride): # for scalar
                y_stride = [0]
            if len(y.shape) == 2:
                arr = _arithmetic.lib.op_2d(x.ptr, y.ptr, x.numel, y.numel, y.shape, x.shape, y.stride, y_stride, len(x.shape), 1)
            elif len(y.shape) == 3:
                arr = _arithmetic.lib.op_3d(x.ptr, y.ptr, x.numel, y.numel, y.shape, x.shape, y.stride, y_stride, len(x.shape), 1)

        return CPU.ffi.gc(arr, _arithmetic.lib.free_op)

    @staticmethod
    @dispatcher.register(BinaryOps.NEG, Device.CPU)
    def neg(x: Buffer) -> CDataPtr:
        arr = _neg.lib.neg(x.ptr, x.numel)

        return CPU.ffi.gc(arr, _neg.lib.free_neg)

    @staticmethod
    @dispatcher.register(BinaryOps.MATMUL, Device.CPU)
    def matmul(x: Buffer, y: Buffer) -> CDataPtr:
        arr = _matmul.lib.matmul(x.ptr, y.ptr, x.shape[0], y.shape[1], y.shape[0], y.stride, x.stride)

        return CPU.ffi.gc(arr, _matmul.lib.free_matmul)

    @staticmethod
    @dispatcher.register(UnaryOps.SUM, Device.CPU)
    def sum(x: Buffer) -> CDataPtr:
        arr = _sum.lib.sum(x.ptr, x.numel)

        return CPU.ffi.gc(arr, _sum.lib.free_sum)
