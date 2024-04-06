# from tinygrad
class DType:
    def __init__(self) -> None:
        pass

class dtypes:
    float32 = DType()
    int32 = DType()

    @staticmethod
    def from_py(data):
        return dtypes.float32 if isinstance(data, float) else dtypes.int32