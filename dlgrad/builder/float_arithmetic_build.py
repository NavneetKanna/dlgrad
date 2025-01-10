import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef(
    "float *op_3d(float *x, float *y, int *xshape, int *xstrides, int *yshape, int *ystrides, int outnumel, int op);\
        float *op_2d(float *x, float *y, int *xshape, int *xstrides, int *yshape, int *ystrides, int outnumel, int op);\
        void free_op(float *ptr);")
ffi.set_source("_arithmetic", f"""
    #include "{root_dir}/src/c/arithmetic.h"
""",
sources=[f'{root_dir}/src/c/arithmetic.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
