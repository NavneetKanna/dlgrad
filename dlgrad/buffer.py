"""
This should contain all buffer related tasks.

"""
from dlgrad.c_code import C
import subprocess
import tempfile
import ctypes
import time
import os
from dlgrad.helpers import get_temp_loc
import atexit

class Buffer:
    rand_temp_file = False

    def __init__(self, data, temp_file_loc) -> None:
        self.data_buffer = data
        self._temp_file_loc = temp_file_loc
        atexit.register(self._cleanup)

    def _cleanup(self):
        if os.path.exists(self._temp_file_loc):
            os.remove(self._temp_file_loc)

    @staticmethod
    def _check_temp_file_exists(name):
        for f in os.listdir(get_temp_loc()):
            if f.startswith(name):
                return f 
        return ''

    @staticmethod
    def create_random_buffer(length: int):
        # ctypes.CDLL was taking the most time when i was compiling the prg everytime this func was called
        # and also there is no need to compile everytime this func was called, hence compiling only once
        # and reading the shared file, and now ctypes.CDLL is fast and is no longer taking time.

        temp_file = Buffer._check_temp_file_exists("rand_buffer") 

        if temp_file:
            rand_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}")
        else:
            prg = C._random_buffer()
            with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix="rand_buffer") as output_file:
                temp_file = str(output_file.name)
                subprocess.check_output(args=['clang', '-o2', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', temp_file], input=prg.encode('utf-8'))
                rand_dll = ctypes.CDLL(temp_file)
        
        rand_dll.create_rand_buffer.argtypes = (ctypes.c_int,)
        rand_dll.create_rand_buffer.restype = ctypes.POINTER(ctypes.c_float) 
        return Buffer(rand_dll.create_rand_buffer(length), temp_file)

    @staticmethod
    def free(data):
        temp_file = Buffer._check_temp_file_exists("free") 
        if temp_file:
            free_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}")
        else:
            prg = C._free()
            with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix="free") as output_file:
                temp_file = str(output_file.name)
                subprocess.check_output(args=['clang', '-o2', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', temp_file], input=prg.encode('utf-8'))
                free_dll = ctypes.CDLL(temp_file)
   
        free_dll.free_buf.argtypes = ctypes.c_void_p,
        free_dll.free_buf.restype = None
        free_dll.free_buf(data) 
        
        if os.path.exists(temp_file):
            os.remove(temp_file) 