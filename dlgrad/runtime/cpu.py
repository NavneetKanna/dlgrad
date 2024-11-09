import _add  # type: ignore
import _neg  # type: ignore
import _uniform  # type: ignore
import _matmul # type: ignore

from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import DType, Scalar
from dlgrad.helpers import BinaryOps, BufferOps, get_y_broadcast_ss, prod_


class CPU:
    """
    Main CPU runtime class which handles the logic of calling the compiled C source files.

    This class uses CFFI (C Foreign Function Interface) to interact with C code.
    """
    ffi = FFI()

    def __init__(self) -> None:
        pass
    
    @staticmethod
    @dispatcher.register(BufferOps.CREATE, Device.CPU)
    def create_buffer_from_scalar(x: Scalar) -> Buffer:
        return Buffer(CPU.ffi.new(f"{DType.get_c_dtype((x))} [1]", [x]))

    @staticmethod
    @dispatcher.register(BufferOps.UNIFORM, Device.CPU)
    def uniform(shape: tuple, low: float, high: float) -> Buffer:
        numel = prod_(shape)
        arr = _uniform.lib.uniform(numel, low, high)

        return Buffer(CPU.ffi.gc(arr, _uniform.lib.free_uniform))
    
    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.CPU)
    def add(x, y):
        y_broad_shape, y_broad_stride = get_y_broadcast_ss(x.metadata.shape, y.metadata.shape, y.metadata.stride)

        arr = _add.lib.add(x.data.ptr, y.data.ptr, x.metadata.numel, x.metadata.shape, y_broad_shape, x.metadata.stride, y_broad_stride, x.metadata.ndim)

        return Buffer(CPU.ffi.gc(arr, _add.lib.free_add))
        
    @staticmethod
    @dispatcher.register(BinaryOps.NEG, Device.CPU)
    def neg(x):
        arr = _neg.lib.neg(x.data.ptr, x.numel)

        return Buffer(CPU.ffi.gc(arr, _neg.lib.free_neg))

    @staticmethod
    @dispatcher.register(BinaryOps.MATMUL, Device.CPU)
    def matmul(x, y):
        print(x.shape)
        print(y.shape)
        print(y.stride)

        arr = _matmul.lib.matmul(x.data.ptr, y.data.ptr, x.shape[0], y.shape[1], y.shape[0], y.stride, x.stride)

        return Buffer(CPU.ffi.gc(arr, _matmul.lib.free_matmul))
