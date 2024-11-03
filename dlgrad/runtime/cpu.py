import _uniform   # type: ignore
import _add # type: ignore
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import DType, Scalar
from dlgrad.helpers import BufferOps, BinaryOps, prod_, get_y_broadcast_ss


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
    def uniform(shape: tuple, **kwargs) -> Buffer:
        numel = prod_(shape)
        arr = _uniform.lib.uniform(numel)

        return Buffer(CPU.ffi.gc(arr, _uniform.lib.free_uniform))
    
    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.CPU)
    def add(x, y):
        y_broad_shape, y_broad_stride = get_y_broadcast_ss(x.metadata.shape, y.metadata.shape, y.metadata.stride)
        
        x_shape_ptr = CPU.ffi.new("int[]", len(x.metadata.shape))
        for i, value in enumerate(x.metadata.shape):
            x_shape_ptr[i] = value
        y_shape_ptr = CPU.ffi.new("int[]", len(y_broad_shape))
        for i, value in enumerate(x.metadata.shape):
            y_shape_ptr[i] = value
        x_stride_ptr = CPU.ffi.new("int[]", len(x.metadata.stride))
        for i, value in enumerate(x.metadata.stride):
            x_stride_ptr[i] = value
        y_stride_ptr = CPU.ffi.new("int[]", len(y_broad_stride))
        for i, value in enumerate(x.metadata.stride):
            y_stride_ptr[i] = value

        _add(x.data.ptr, y.data.ptr, x.metadata.numel, x_shape_ptr, y_shape_ptr, x_stride_ptr, y_stride_ptr, x.metadata.ndim)
        