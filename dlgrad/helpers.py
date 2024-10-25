from enum import Enum, auto
from math import prod
from typing import Iterable


class BufferOps(Enum):
    CREATE = auto()
    UNIFORM = auto()


def prod_(x: Iterable) -> int:
    return prod(x)