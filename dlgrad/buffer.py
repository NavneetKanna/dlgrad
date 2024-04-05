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


class Buffer:
    rand_temp_file = False

    def __init__(self, data) -> None:
        self.data_buffer = data

    @staticmethod
    def create_random_buffer(length: int):
        prg = C._random_buffer()
        # ctypes.CDLL was taking the most time when i was compiling the prg everytime this func was called
        # and also there is no need to compile everytime this func was called, hence compiling only once
        # and reading the shared file, and now ctypes.CDLL is fast and is no longer taking time.
        for f in os.listdir(get_temp_loc()):
            print(f)
            if f.startswith("rand_buffer"):
                s = time.perf_counter()
                rand_dll = ctypes.CDLL(f"{get_temp_loc()}/{f}")
                e = time.perf_counter()
                print(f"time to open cdll {e-s:.4f}s")
            else:
                with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix="rand_buffer") as output_file:
                    print(str(output_file.name))
                    subprocess.check_output(args=['clang', '-o2', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', str(output_file.name)], input=prg.encode('utf-8'))
                    s = time.perf_counter()
                    rand_dll = ctypes.CDLL(str(output_file.name))
                    e = time.perf_counter()
                    print(f"time to open cdll {e-s:.4f}s")

            rand_dll.create_rand_buffer.argtypes = (ctypes.c_int,)
            rand_dll.create_rand_buffer.restype = ctypes.POINTER(ctypes.c_float) 
            return Buffer(rand_dll.create_rand_buffer(length))

    @staticmethod
    def free(data):
        prg = C._free()
        with tempfile.NamedTemporaryFile(delete=True) as output_file:
            subprocess.check_output(args=['clang', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', str(output_file.name)], input=prg.encode('utf-8'))
            free_dll = ctypes.CDLL(str(output_file.name))
            free_dll.free_buf.argtypes = ctypes.c_void_p,
            free_dll.free_buf.restype = None
            free_dll.free_buf(data) 