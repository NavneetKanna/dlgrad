from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.helpers import BufferOps


class CPU:
    ffi = FFI()

    def __init__(self) -> None:
        pass
    
    @staticmethod
    @dispatcher.register(BufferOps.CREATE, Device.CPU)
    def create_buffer_from_scalar(x) -> Buffer:
        return Buffer(CPU.ffi.new("int [1]", [x]))