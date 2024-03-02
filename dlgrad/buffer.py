from typing import Union
import ctypes as ct
# from dlgrad.helpers import get_list_dim

dll = ct.CDLL('./backend/c/test.so')
dll.create_rand_buffer.argtypes = ct.c_int, ct.c_int, ct.c_int, ct.c_int 
dll.create_rand_buffer.restype = ct.POINTER(ct.c_float)  

class Buffer:
    """
    C buffer is not created for scalar data and n dim list 
    
    """

    def __init__(self, data: Union[int, float, list]):
        self._data = data
    
    @staticmethod
    def create_scalar_buffer(data: Union[int, float]): return Buffer(data) 
    
    @staticmethod
    def create_list_buffer(data: list): return Buffer(data)

    @staticmethod
    def create_rand_buffer(shape): 
        l = len(shape)
        if l == 1: dll.create_rand_buffer(0, 0, 0, shape[0])
        if l == 2: dll.create_rand_buffer(0, 0, shape[0], shape[1])
        if l == 3: dll.create_rand_buffer(0, shape[0], shape[1], shape[2])
        if l == 4: dll.create_rand_buffer(shape[0], shape[1], shape[2], shape[3])
