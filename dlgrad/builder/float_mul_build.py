import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("float *mul_with_scalar(float *x, float *y, int xnumel); \
          float *mul_with_dim1(float *x, float *y, int xnumel, int at); \
          float *mul_with_dim0(float *x, float *y, int xnumel, int ynumel, int at); \
          float *mul(float *x, float *y, int xnumel); \
          float *mul_3d_with_2d(float *x, float *y, int xnumel, int ynumel); \
          float *mul_with_dim1_with_dim0(float *x, float *y, int xnumel, int ynumel, int at, int ncols); \
          void free_mul(float *ptr);")
ffi.set_source("_mul", f"""
    #include "{root_dir}/src/c/mul.h"
""",
sources=[f'{root_dir}/src/c/mul.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
