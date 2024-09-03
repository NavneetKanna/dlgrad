# idea from tinygrad
from dataclasses import dataclass
import ctypes

@dataclass
class DType:
    name: str

    def __repr__(self) -> str:
        return f"dlgrad.{self.name}"


class dtypes:
    float32 = DType("float32")
    int32 = DType("int32")

    @staticmethod
    def from_py(data):
        return dtypes.float32 
        # return dtypes.float32 if isinstance(data, float) else dtypes.int32

    @staticmethod
    def get_c_dtype(dtype, map_ctype=False):
        if dtype == dtypes.float32:
            return ctypes.c_float if map_ctype else "float"
