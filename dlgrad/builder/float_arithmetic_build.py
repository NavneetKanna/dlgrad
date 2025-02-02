import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

cdef = """
void op_3d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op);
void op_2d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op);
void with_scalar(float *x, float *out, float y, int xnumel, int op);
"""
ffi.cdef(cdef)

ffi.set_source("_arithmetic", f"""
        #include "{root_dir}/src/c/arithmetic.h"
    """,
    sources=[f'{root_dir}/src/c/arithmetic.c'],
    extra_compile_args=["-O2", "-march=native"]
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
