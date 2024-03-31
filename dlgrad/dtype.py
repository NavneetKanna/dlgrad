from typing import Final

class dtypes:
    float32: Final = None
    float32_ptr: Final = None
    int32: Final = None

    @staticmethod
    def from_py(data):
        return dtypes.float32 if isinstance(data, float) else dtypes.int32