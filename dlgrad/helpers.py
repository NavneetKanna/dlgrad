import itertools
from enum import Enum, auto
from math import prod
from typing import Iterable

from cffi import FFI

ffi = FFI()


class BufferOps(Enum):
    CREATE = auto()
    UNIFORM = auto()
    ARANGE = auto()

class BinaryOps(Enum):
    ADD = auto()

def prod_(x: Iterable) -> int:
    return prod(x)

def get_y_broadcast_ss(x_shape: tuple, y_shape: tuple, ystride: tuple) -> tuple:
    """
    Compute the broadcast shape and stride of y Tensor. Pad with 1's until the dimensions match.
    It is assumed that x is the higher dimension Tensor or the shape y has to be broadcasted to.

    Parameters:
        x_shape(tuple): The x Tensor shape.
        y_shape (tuple): The y Tensor shape.
        y_stride (tuple): The y Tebsir stride.

    Returns:
        tuple: The y Tensor's broadcasted shape.

    Raises:
        AssertionError: If the shapes are incompatible.
    """
    y_broad_shape = []
    ystride: list = list(ystride)[::-1]
    for i, j in itertools.zip_longest(reversed(x_shape), reversed(y_shape)):
        if i is not None and j is not None and i != j and i != 1 and j != 1:
            raise AssertionError(f"{i} and {j} dim does not match")

        if i == j:
            y_broad_shape.append(i)
        elif j is None or j == 1:
            y_broad_shape.append(1)
            ystride.append(1)

    return tuple(reversed(y_broad_shape)), tuple(reversed(ystride))

def calculate_stride(shape: tuple) -> tuple:
    stride = []
    stride_value = 1
    for dim in reversed(shape):
        stride.append(stride_value)
        stride_value *= dim
    return tuple(reversed(stride))
