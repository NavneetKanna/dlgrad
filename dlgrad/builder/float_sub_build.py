import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("float *sub_with_scalar(float *x, float *y, int xnumel); \
          float *sub_with_dim1(float *x, float *y, int xnumel, int at); \
          float *sub_with_dim0(float *x, float *y, int xnumel, int ynumel, int at); \
          float *sub(float *x, float *y, int xnumel); \
          float *sub_3d_with_2d(float *x, float *y, int xnumel, int ynumel); \
          float *sub_with_dim1_with_dim0(float *x, float *y, int xnumel, int ynumel, int at, int ncols); \
          void free_sub(float *ptr);")
ffi.set_source("_sub", f"""
    #include "{root_dir}/src/c/sub.h"
""",
sources=[f'{root_dir}/src/c/sub.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
