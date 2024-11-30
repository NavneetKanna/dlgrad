import itertools
from enum import Enum, auto
from math import prod
from typing import Iterable

from cffi import FFI

ffi = FFI()


class UnaryOps(Enum):
    SUM = auto()
    TRANSPOSE = auto()

class BufferOps(Enum):
    CREATE = auto()
    UNIFORM = auto()
    ARANGE = auto()
    FULL = auto()

class BinaryOps(Enum):
    ADD = auto()
    SUB = auto()
    NEG = auto()
    MATMUL = auto()

def prod_(x: Iterable) -> int:
    return prod(x) if x else tuple()

def check_broadcast(x_shape: tuple, y_shape: tuple) -> bool:
    """
    Check if y_shape is broadcastable to x_shape.
    It is assumed that x is the higher dimension Tensor.

    Parameters:
        x_shape (tuple): The x Tensor shape.
        y_shape (tuple): The y Tensor shape.

    Returns:
        bool: True if they are broadcastable. 

    Raises:
        AssertionError: If the shapes are not broadcastable.
    """
    for i, j in itertools.zip_longest(reversed(x_shape), reversed(y_shape)):
        if i is not None and j is not None and i != j and i != 1 and j != 1:
            raise AssertionError(f"Cannot broadcast {y_shape} to {x_shape}, the dimensions {i} and {j} dont match")

    return True

def get_brodcast_tensor(x, y):
    if len(x.shape) > len(y.shape):
        return x, y
    elif len(x.shape) < len(y.shape):
        return y, x
    else:
        for dim_x, dim_y in zip(x.shape, y.shape):
            if dim_x > dim_y:
                return x, y
            elif dim_x < dim_y:
                return y, x

        return x, y
    
def calculate_stride(shape: tuple|int) -> tuple:
    if not shape:
        return tuple()

    if isinstance(shape, int):
        return (1,)

    stride = []
    stride_value = 1
    for dim in reversed(shape):
        stride.append(stride_value)
        stride_value *= dim

    return tuple(reversed(stride))
