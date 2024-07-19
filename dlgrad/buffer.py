"""
This should contain all buffer related tasks.

"""
from __future__ import annotations
from dlgrad.c_code import C
import subprocess
import tempfile
import ctypes
import os
from dlgrad.helpers import get_temp_loc, check_temp_file_exists
import atexit

class Buffer:
    def __init__(self, data, temp_file_loc: str = '') -> None:
        self._buffer = data

        if temp_file_loc:
            self._temp_file_loc = temp_file_loc
            atexit.register(self._cleanup)

    def _cleanup(self):
        if os.path.exists(self._temp_file_loc):
            os.remove(self._temp_file_loc)

    # TODO: Move to cpu
    @staticmethod
    def uniform(length: int, low=0.0, high=1.0) -> Buffer:
        # ctypes.CDLL was taking the most time when i was compiling the prg everytime this func was called
        # and also there is no need to compile everytime this func is called, hence compiling only once
        # and reading the shared file, ctypes.CDLL is faster and is no longer taking time, although, the 
        # first time it is long.

        temp_file = check_temp_file_exists(starts_with="rand_buffer") 
        if temp_file:
            temp_file = f"{get_temp_loc()}/{temp_file}"
            rand_dll = ctypes.CDLL(temp_file)
        else:
            prg = C._random_buffer()
            with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix="rand_buffer") as output_file:
                temp_file = str(output_file.name)
                subprocess.check_output(args=['clang', '-O2', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', temp_file], input=prg.encode('utf-8'))
                rand_dll = ctypes.CDLL(temp_file)
        
        rand_dll.create_rand_buffer.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_float)
        rand_dll.create_rand_buffer.restype = ctypes.POINTER(ctypes.c_float) 
        data = rand_dll.create_rand_buffer(length, low, high)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")
        return Buffer(data, temp_file)

    @staticmethod
    def ones(length: int) -> Buffer:
        temp_file = check_temp_file_exists(starts_with="ones_buffer") 
        if temp_file:
            temp_file = f"{get_temp_loc()}/{temp_file}"
            ones_dll = ctypes.CDLL(temp_file)
        else:
            prg = C._ones_buffer()
            with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix="ones_buffer") as output_file:
                temp_file = str(output_file.name)
                subprocess.check_output(args=['clang', '-O2', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', temp_file], input=prg.encode('utf-8'))
                ones_dll = ctypes.CDLL(temp_file)
        
        ones_dll.create_ones_buffer.argtypes = (ctypes.c_int,)
        ones_dll.create_ones_buffer.restype = ctypes.POINTER(ctypes.c_float) 
        data = ones_dll.create_ones_buffer(length)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")
        return Buffer(data, temp_file)

    @staticmethod
    def free(data) -> None:
        temp_file = check_temp_file_exists(starts_with="free") 
        if temp_file:
            temp_file = f"{get_temp_loc()}/{temp_file}"
            free_dll = ctypes.CDLL(temp_file)
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