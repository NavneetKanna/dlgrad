from enum import Enum, auto
from math import prod
from typing import Iterable
import itertools

from cffi import FFI

ffi  = FFI()


class BufferOps(Enum):
    CREATE = auto()
    UNIFORM = auto()

class BinaryOps(Enum):
    ADD = auto()

def prod_(x: Iterable) -> int:
    return prod(x)

def get_broadcast_shape(shape1: tuple, shape2: tuple):
    """
    Compute the broadcast shape given two shapes.

    Parameters:
        shape1 (tuple): The first shape.
        shape2 (tuple): The second shape.

    Returns:
        tuple: The broadcast shape.

    Raises:
        AssertionError: If the shapes are incompatible.
    """
    out_shape = []
    for i, j in itertools.zip_longest(reversed(shape1), reversed(shape2)):
        if i is not None and j is not None and i != j:
            raise AssertionError(f"{i} and {j} dim does not match")
        out_shape.append(i if i is not None else j)

    return tuple(reversed(out_shape))
