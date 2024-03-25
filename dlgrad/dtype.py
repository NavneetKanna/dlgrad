from typing import Final

class dtypes:
    float32: Final
    float32_ptr: Final
    int32: Final

    @staticmethod
    def from_py(data):
        return dtypes.float32 if isinstance(data, float) else dtypes.int32