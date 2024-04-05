"""
This should contain all buffer related tasks.

"""
from dlgrad.c_code import C
import subprocess
import tempfile
import ctypes


# TODO: cdll is taking the most time
class Buffer:
    _rand_dll = None

    def __init__(self, data) -> None:
        self.data_buffer = data


    @staticmethod
    def create_random_buffer(length: int):
        prg = C._random_buffer()
        with tempfile.NamedTemporaryFile(delete=True) as output_file:
            subprocess.check_output(args=['clang', '-o2', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', str(output_file.name)], input=prg.encode('utf-8'))

            print(Buffer._rand_dll)
            
            rand_dll = ctypes.CDLL(str(output_file.name))
            Buffer._rand_dll = rand_dll
            rand_dll.create_rand_buffer.argtypes = (ctypes.c_int,)
            rand_dll.create_rand_buffer.restype = ctypes.POINTER(ctypes.c_float) 

            return Buffer(rand_dll.create_rand_buffer(length))

    @staticmethod
    def free(data):
        prg = C._free()
        with tempfile.NamedTemporaryFile(delete=True) as output_file:
            subprocess.check_output(args=['clang', '-x', 'c', '-', '-shared', '-o', str(output_file.name)], input=prg.encode('utf-8'))
            free_dll = ctypes.CDLL(str(output_file.name))
            free_dll.free_buf.argtypes = ctypes.c_void_p,
            free_dll.free_buf.restype = None
            free_dll.free_buf(data) 