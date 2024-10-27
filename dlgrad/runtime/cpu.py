import random

import _uni
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import DType, Scalar
from dlgrad.helpers import BufferOps, prod_


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
        arr = _uni.lib.uni(numel)

        return Buffer(arr)
        