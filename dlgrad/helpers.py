from enum import Enum, auto
from math import prod
from typing import Iterable
import itertools


class BufferOps(Enum):
    CREATE = auto()
    UNIFORM = auto()

class BinaryOps(Enum):
    ADD = auto()

def prod_(x: Iterable) -> int:
    return prod(x)

def get_y_broadcast_shape(x_shape: tuple, y_shape: tuple) -> tuple:
    """
    Compute the broadcast shape of y Tensor. Pad with 1's until the dimensions match.
    It is assumed that x is the higher dimension Tensor or the shape y has to be broadcasted to.

    Parameters:
        shape1 (tuple): The x Tensor shape.
        shape2 (tuple): The y Tensor shape.

    Returns:
        tuple: The y Tensor's broadcasted shape.

    Raises:
        AssertionError: If the shapes are incompatible.
    """
    y_broad_shape = []
    for i, j in itertools.zip_longest(reversed(x_shape), reversed(y_shape)):
        if i is not None and j is not None and i != j and i != 1 and j != 1:
            raise AssertionError(f"{i} and {j} dim does not match")

        if i == j:
            y_broad_shape.append(i)
        elif j is None or j == 1:
            y_broad_shape.append(1)

    return tuple(reversed(y_broad_shape))
