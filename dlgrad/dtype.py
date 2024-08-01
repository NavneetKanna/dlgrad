# idea from tinygrad
from dataclasses import dataclass


@dataclass
class DType:
    name: str
    def __repr__(self) -> str:
        return f"dlgrad.{self.name}"

class dtypes:
    float32 = DType('float32')
    int32 = DType('int32')

    @staticmethod
    def from_py(data):
        return dtypes.float32 if isinstance(data, float) else dtypes.int32
    
    @staticmethod
    def get_c_dtype(dtype):
        if dtype == dtypes.float32: 
            return 'float' 