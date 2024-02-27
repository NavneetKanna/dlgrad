from typing import Union
import ctypes as ct
# from dlgrad.helpers import get_list_dim


dll = ct.CDLL('../backend/c/test.so')
dll.create_rand_buffer.argtypes = ct.POINTER(ct.c_int), ct.c_int
# dll.func.restype = ct.POINTER(ct.c_char)
# dll.freeMem.argtypes = ct.c_void_p,
# dll.freeMem.restype = None


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

    #TODO: Restrict the dim to 4, so things become easier
    @staticmethod
    def create_rand_buffer(shape): 

        dll.create_rand_buffer()
