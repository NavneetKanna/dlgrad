import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("float *add_with_scalar(float *x, float *y, int xnumel); \
          float *add_with_dim1(float *x, float *y, int xnumel, int at); \
          float *add_with_dim0(float *x, float *y, int xnumel, int ynumel, int at); \
          float *add(float *x, float *y, int xnumel); \
          float *add_3d_with_2d(float *x, float *y, int xnumel, int ynumel); \
          float *add_with_dim1_with_dim0(float *x, float *y, int xnumel, int ynumel, int at, int ncols); \
          void free_add(float *ptr);")
ffi.set_source("_add", f"""
    #include "{root_dir}/src/c/add.h"
""",
sources=[f'{root_dir}/src/c/add.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
