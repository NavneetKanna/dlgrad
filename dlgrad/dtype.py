from enum import Enum, auto

Scalar = int | float

class DType(Enum):
    FLOAT32 = auto()

    @staticmethod
    def from_str(d: str):
        try:
            return DType[d.upper()]
        except KeyError:
            print(f"Invalid dtype: {d}")

    @staticmethod
    def get_c_dtype(d: Enum) -> str:
        pass
