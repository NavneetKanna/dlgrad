import itertools
from collections.abc import Iterable
from enum import Enum, auto
from math import prod

from cffi import FFI

ffi = FFI()


class UnaryOps(Enum):
    SUM = auto()
    MAX = auto()
    TRANSPOSE = auto()
    RELU = auto()

class BufferOps(Enum):
    CREATE = auto()
    UNIFORM = auto()
    ARANGE = auto()
    FULL = auto()

class BinaryOps(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    NEG = auto()
    MATMUL = auto()
    GT = auto() # >

class CustomOps(Enum):
    INDEX = auto()


def prod_(x: Iterable) -> int:
    return prod(x) if x else 1

def check_broadcast(x_shape: tuple, y_shape: tuple) -> bool:
    """
    Check if shapes are broadcastable.

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
            raise AssertionError(f"Cannot broadcast {y_shape} to {x_shape}, the dimensions {i} and {j} dont match")  # noqa: E501

    return True

def get_brodcast_tensor(x, y):  # noqa: ANN001, ANN201
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

def resolve_ndim(inp_shape: tuple, grad_shape: tuple) -> int:
    if not check_broadcast(x_shape=inp_shape, y_shape=grad_shape):
        raise AssertionError(f"Cannot reduce grad of shape {grad_shape} to the input shape {inp_shape}")  # noqa: E501

    if inp_shape == grad_shape:
        return 0

    ndim = 0
    dim = len(max(inp_shape, grad_shape)) - 1
    for i, j in itertools.zip_longest(reversed(inp_shape), reversed(grad_shape)):
        dim -= 1
        if i != j:
            ndim += 1

    return ndim

def cal_sum_out_shape(ndim: int, dim: int, inp_shape: tuple) -> tuple:
    out_shape = tuple()
    if ndim == 3:
        if dim == 0:
            out_shape = (inp_shape[1], inp_shape[2])
        elif dim == 1:
            out_shape = (inp_shape[0], inp_shape[2])
        elif dim == 2:
            out_shape = (inp_shape[0], inp_shape[1])
        else:
            out_shape = tuple()
    elif ndim == 2:
        if dim == 0:
            out_shape = (inp_shape[1],)
        elif dim ==1:
            out_shape = (inp_shape[0],)
        else:
            out_shape = tuple()

    return out_shape
