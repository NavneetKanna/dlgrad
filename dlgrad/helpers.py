from enum import Enum, auto
from math import prod
from typing import Iterable
import os 
from cffi import FFI


ffi  = FFI()

root_dir = os.path.dirname(os.path.abspath(__file__))

class BufferOps(Enum):
    CREATE = auto()
    UNIFORM = auto()


def prod_(x: Iterable) -> int:
    return prod(x)