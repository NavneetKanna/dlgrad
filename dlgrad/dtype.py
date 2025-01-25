from __future__ import annotations

from enum import Enum, auto
from typing import NewType

from cffi import FFI

CDataPtr = NewType("CDataPtr", FFI().CData)
Scalar = float

class DType(Enum):
    FLOAT32 = auto()

    @staticmethod
    def from_str(d: str) -> str:
        try:
            return DType[d.upper()]
        except KeyError:
            raise ValueError(f"Invalid dtype: {d}")

    @classmethod
    def _get_n_bytes(cls) -> dict:
        return {cls.FLOAT32: 4}

    @staticmethod
    def get_n_bytes(d: DType) -> int:
        return DType._get_n_bytes()[d]
