from enum import Enum, auto
from math import prod
from typing import Iterable
from cffi import FFI


ffi  = FFI()


class BufferOps(Enum):
    CREATE = auto()
    UNIFORM = auto()


def prod_(x: Iterable) -> int:
    return prod(x)