import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("void matmul(float *x, float *y, float *out, int x_rows, int y_cols, int y_rows, int *ystride, int *xstride);")
ffi.set_source("_matmul", f"""
    #include "{root_dir}/src/c/matmul.h"
""",
sources=[f'{root_dir}/src/c/matmul.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
