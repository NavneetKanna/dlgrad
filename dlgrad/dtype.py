from __future__ import annotations

from enum import Enum, auto

Scalar = int | float

class DType(Enum):
    FLOAT32 = auto()
    
    @staticmethod
    def from_str(d: str):
        try:
            return DType[d.upper()]
        except KeyError:
            raise ValueError(f"Invalid dtype: {d}")

    @staticmethod
    def get_dtype_from_py(d: Scalar) -> DType:
        if isinstance(d, int):
            return DType.INT32
        elif isinstance(d, float):
            return DType.FLOAT32
    
    @staticmethod
    def get_c_dtype(d: Scalar) -> str:
        if isinstance(d, float):
            return "float"

    @classmethod
    def _get_n_bytes(cls):
        return {cls.FLOAT32: 4}
    
    @staticmethod
    def get_n_bytes(d: DType) -> int:
        return DType._get_n_bytes()[d]
