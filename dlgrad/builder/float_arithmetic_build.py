import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef(
    "void op_3d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op);\
       void op_2d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op);\
        void add_with_1d(float *x, float *y, float *out, int xnumel, int ynumel, int op);")
ffi.set_source("_arithmetic", f"""
    #include "{root_dir}/src/c/arithmetic.h"
""",
sources=[f'{root_dir}/src/c/arithmetic.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
