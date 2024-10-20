from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.helpers import BufferOps
from dlgrad.dtype import Scalar


class CPU:
    """
    The main CPU runtime class which handles the logic of calling the compiled
    C source files.

    cffi is used to interact with the C code.
    """
    ffi = FFI()

    def __init__(self) -> None:
        pass
    
    @staticmethod
    @dispatcher.register(BufferOps.CREATE, Device.CPU)
    def create_buffer_from_scalar(x: Scalar) -> Buffer:
        return Buffer(CPU.ffi.new(f"{type(x)} [1]", [x]))