from enum import Enum, auto


class DType(Enum):
    FLOAT32 = auto()

    @staticmethod
    def from_str(d: str):
        try:
            return DType[d.upper()]
        except KeyError:
            print(f"Invalid dtype: {d}")
