import _add  # type: ignore
import _full  # type: ignore
import _matmul  # type: ignore
import _neg  # type: ignore
import _sum  # type: ignore
import _uniform  # type: ignore
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import DType, Scalar, CDataPtr
from dlgrad.helpers import BinaryOps, BufferOps, UnaryOps, prod_


# TODO: Calling ffi.gc() twice one after the other leads to error, find alternative
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
        return Buffer(CPU.ffi.new(f"{DType.get_c_dtype((x))} [1]", [x]), tuple(), device=Device.CPU, ndim=0)

    @staticmethod
    @dispatcher.register(BufferOps.UNIFORM, Device.CPU)
    def uniform(shape: tuple, low: float, high: float) -> Buffer:
        # numel = prod_(shape) if isinstance(shape, tuple) else shape
        numel = prod_(shape)
        arr = _uniform.lib.uniform(numel, low, high)

        return Buffer(CPU.ffi.gc(arr, _uniform.lib.free_uniform), shape, device=Device.CPU)
    
    @staticmethod
    @dispatcher.register(BufferOps.FULL, Device.CPU)
    def full(shape: tuple, fill_value: Scalar) -> Buffer:
        arr = _full.lib.full(prod_(shape), fill_value)

        return Buffer(CPU.ffi.gc(arr, _full.lib.free_full), shape, device=Device.CPU)

    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.CPU)
    def add(x: Buffer, y: Buffer) -> CDataPtr:
        if not (y_stride := y.stride): # for scalar
            y_stride = [0]

        if len(x.shape) == 2:
            arr = _add.lib.add_2d(x.ptr, y.ptr, x.numel, x.shape, y.shape, x.stride, y_stride, len(y.shape))
        elif len(x.shape) == 3:
            arr = _add.lib.add_3d(x.ptr, y.ptr, x.numel, x.shape, y.shape, x.stride, y.stride, len(y.shape))

        return CPU.ffi.gc(arr, _add.lib.free_add)
        
    @staticmethod
    @dispatcher.register(BinaryOps.NEG, Device.CPU)
    def neg(x: Buffer) -> CDataPtr:
        arr = _neg.lib.neg(x.ptr, x.numel)

        return CPU.ffi.gc(arr, _neg.lib.free_neg)

    @staticmethod
    @dispatcher.register(BinaryOps.MATMUL, Device.CPU)
    def matmul(x: Buffer, y: Buffer) -> Buffer:
        arr = _matmul.lib.matmul(x.ptr, y.ptr, x.shape[0], y.shape[1], y.shape[0], y.stride, x.stride)

        return Buffer(CPU.ffi.gc(arr, _matmul.lib.free_matmul), (x.shape[0], y.shape[1]), device=Device.CPU)

    @staticmethod
    @dispatcher.register(UnaryOps.SUM, Device.CPU)
    def sum(x: Buffer) -> CDataPtr:
        arr = _sum.lib.sum(x.ptr, x.numel)

        return CPU.ffi.gc(arr, _sum.lib.free_sum)

    # TODO: Check if x is del, then even the transposed is del
    @staticmethod
    @dispatcher.register(UnaryOps.TRANSPOSE, Device.CPU)
    def transpose(x: Buffer) -> Buffer:
        return Buffer(x.ptr, x.shape[::-1], stride=x.stride[::-1], device=Device.CPU)