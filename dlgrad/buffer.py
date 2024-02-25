from typing import Union
import ctypes
from dlgrad.helpers import get_list_dim

class Buffer:
    def __init__(self, data):
        self.data = data
    
    @staticmethod
    def create_scalar_buffer(data):
        return Buffer(data) 
    