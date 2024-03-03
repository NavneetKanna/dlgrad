from typing import Union
import ctypes as ct


class c_rand_buffer:
    dll = ct.CDLL('./backend/c/test.so')
    dll.create_rand_buffer.argtypes = (ct.c_int,)
    dll.create_rand_buffer.restype = ct.POINTER(ct.c_float)  
    dll.free_buf.argtypes = ct.c_void_p,
    dll.free_buf.restype = None

    @staticmethod
    def _create(size): return c_rand_buffer.dll.create_rand_buffer(size)
        
    @staticmethod
    def _free(rand_buf_p): c_rand_buffer.dll.free_buf(rand_buf_p)
