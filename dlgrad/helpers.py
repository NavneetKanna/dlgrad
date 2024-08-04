import os
from enum import Enum, auto

GRAPH = os.getenv("GRAPH", 0)

def get_graph():
    return GRAPH

def set_graph(val):
    global GRAPH
    GRAPH = val

class BinaryOps(Enum):
    ADD = auto()
    MATMUL = auto()

class UnaryOps(Enum):
    TRANSPOSE = auto()
    SUM = auto()

class BufferOps(Enum):
    UNIFORM = auto()
    ONES = auto()

class Device(Enum):
    CPU = auto()
    GPU = auto()

class ShapeError(Exception): ...
class IndexError(Exception): ...

class BroadcastHelper:
    out_len = 0

def calculate_axis(shape1: tuple, shape2: tuple) -> int:
    if shape1 == shape2: 
        return -1
    elif shape1[0] == shape2[0]: 
        return 0
    else: 
        return 1

def calculate_numel(shape: tuple):
    out_len = 1
    for i in shape:
        out_len *= i
    return out_len

# TODO: check win or unix
def get_temp_loc():
    return "/tmp"

def get_list_dim(data: list, dim=0):
    if isinstance(data, list): 
        dim +=1
        return get_list_dim(data[0], dim)
    else: 
        return dim

def calculate_stride(shape: tuple):
    if len(shape) == 1:
        return [1]

    stride = [i for i in range(len(shape))]
    
    if len(shape) == 2:
        stride[0] = shape[-1]
        stride[1] = 1
        return stride
    
    for i in range(len(shape) - 2):
        prod = 1
        for j in range(i+1, len(shape)):
            prod *= shape[j]
        stride[i] = prod
    stride[-2] = shape[-1]
    stride[-1] = 1
    
    return stride

# https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html
def calculate_nchw_offset(n=0, c=0, h=0, w=0, N=0, C=0, H=0): 
    return (n * N) + (c * C) + (h * H) + w 

def check_temp_file_exists(starts_with: str) -> str:
    for f in os.listdir(get_temp_loc()):
        if f.startswith(starts_with):
            return f 
    return ''
