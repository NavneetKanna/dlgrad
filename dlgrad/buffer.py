from typing import Union
import ctypes
from dlgrad.helpers import get_list_dim

class Buffer:
    """
    C buffer is not created for scalar data and for arrays/list of dim 1
    
    """

    def __init__(self, data: Union[int, float, list]):
        self._data = data
    
    @staticmethod
    def create_scalar_buffer(data: Union[int, float]):
        return Buffer(data) 
    
    @staticmethod
    def create_list_buffer(data: list):
        dim = get_list_dim(data)
        if dim == 1: return Buffer(data)
        else: pass