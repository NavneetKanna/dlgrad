"""
This should contain all buffer related tasks.

"""
from dlgrad.c_code import C
import subprocess
import tempfile
import ctypes

class Buffer:
    def __init__(self) -> None:
        pass

    def create_random_buffer(self, length: int):
        prg = C._random_buffer(length)
        with tempfile.NamedTemporaryFile(delete=True) as output_file:
            subprocess.check_output(args=['clang', '-x', 'c', '-', '-shared', '-o', str(output_file.name)], input=prg.encode('utf-8'))
            rand_dll = ctypes.CDLL(str(output_file.name))
            rand_dll.create_rand_buffer.argtypes = (ctypes.c_int,)
            rand_dll.create_rand_buffer.restype = ctypes.POINTER(ctypes.c_float) 

