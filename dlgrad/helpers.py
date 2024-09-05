import os
from enum import Enum, auto
from typing import Optional

GRAPH = os.getenv("GRAPH")


def get_graph():
    return GRAPH

def set_graph(val):
    global GRAPH
    GRAPH = val


class BinaryOps(Enum):
    ADD = auto()
    DIV = auto()
    SUB = auto()
    MATMUL = auto()


class UnaryOps(Enum):
    TRANSPOSE = auto()
    SUM = auto()
    MAX = auto()
    EXP = auto()
    LOG = auto()
    NEG = auto()


class BufferOps(Enum):
    UNIFORM = auto()
    ONES = auto()
    CUSTOM = auto()


class Device(Enum):
    CPU = auto()
    GPU = auto()


class ShapeError(Exception): ...
class AllocationError(Exception): ...


def calculate_sum_axis(shape1: tuple, shape2: tuple) -> int:
    if shape1[0] == shape2[0]:
        return 1
    return 0

def calculate_add_axis(shape1: tuple, shape2: tuple) -> Optional[int]:
    if not shape1 or not shape2:
        return -2
    if shape1 == shape2:
        return -1
    if shape1[0] == shape2[0]:
        return 0
    if shape1[-1] == shape2[-1]:
        return 1
    return None

def get_broadcast_shape(x: "Tensor", y: "Tensor"): # noqa: F821 # type: ignore
        shape1 = x.shape
        shape2 = y.shape

        if x.ndim > 2 or y.ndim > 2 and shape1 != shape2:
            print("dlgrad does not support broadcasting for dims greater than 2")

        output_shape = []

        shape1 = shape1[::-1]
        shape2 = shape2[::-1]

        for i in range(max(len(shape1), len(shape2))):
            dim1 = shape1[i] if i < len(shape1) else 1
            dim2 = shape2[i] if i < len(shape2) else 1
            if dim1 == 1 or dim2 == 1 or dim1 == dim2:
                output_shape.append(max(dim1, dim2))
            else:
                # TODO: Add error here
                print("Shapes are not compatible for broadcasting")

        return tuple(output_shape[::-1])

def flatten(x):
    result = []
    for item in x:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def analyse_list(x: list):
    out_shape = (len(x),)
    ndim = 1
    
    if isinstance(x[0], list):
        if not len(set(map(len, x))) == 1:
            # TODO: raise error
            print("all len should be equal")
        # get dim
        shape = []
        lst = x
        while isinstance(lst, list):
            shape.append(len(lst))
            lst = lst[0] if lst else []
        out_shape = tuple(shape)
        ndim = len(shape)
        
        x = flatten(x) # flatten into 1d
    
    x = [float(i) for i in x] # does this help when converting to float in c ?

    return prod(out_shape), out_shape, ndim

def calculate_numel(shape: tuple):
    out_len = 1
    for i in shape:
        out_len *= i
    return out_len

def prod(x):
    if isinstance(x, tuple):
        o = 1
        for i in x: 
            o *= i
        return o
    
    return x

def calculate_uops(shape1, axis, keepdim=None):
    # for unary ops
    if axis is None:
        return (), 1, 1, ()
        
    out_shape = shape1[:axis] + shape1[axis+1:] if not keepdim else shape1[:axis] + (1,) + shape1[axis+1:]
    numel = calculate_numel(out_shape)
    ndim = len(out_shape)
    stride = calculate_stride(out_shape)
    return out_shape, numel, ndim, stride

def calculate_stride(shape: tuple):
    if len(shape) == 1:
        return [1]

    stride = list(range(len(shape)))

    if len(shape) == 2:
        stride[0] = shape[-1]
        stride[1] = 1
        return stride

    for i in range(len(shape) - 2):
        prod = 1
        for j in range(i + 1, len(shape)):
            prod *= shape[j]
        stride[i] = prod
    stride[-2] = shape[-1]
    stride[-1] = 1

    return stride

# https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html
def calculate_nchw_offset(n=0, c=0, h=0, w=0, N=0, C=0, H=0):
    return (n * N) + (c * C) + (h * H) + w

def get_temp_loc():
    return "/tmp"

def check_temp_file_exists(starts_with: str) -> str:
    for f in os.listdir(get_temp_loc()):
        if f.startswith(starts_with):
            return f
    return ""

def get_shared_lib_name(name: str, dtype: str = '', device: str = '') -> str:
    # TODO: Check if mac or linux
    return f"{get_temp_loc()}/{name}_{dtype}_{device}.dylib"