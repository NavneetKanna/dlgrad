from __future__ import annotations

from enum import Enum, auto

Scalar = int | float

class DType(Enum):
    FLOAT32 = auto()
    # INT32 = auto()

    @staticmethod
    def from_str(d: str):
        try:
            return DType[d.upper()]
        except KeyError:
            print(f"Invalid dtype: {d}")

    @staticmethod
    def get_dtype_from_py(d: Scalar) -> DType:
        if isinstance(d, int):
            return DType.INT32
        elif isinstance(d, float):
            return DType.FLOAT32
    
    @staticmethod
    def get_c_dtype(d: Scalar) -> str:
        if isinstance(d, int):
            return "int"
        elif isinstance(d, float):
            return "float"
