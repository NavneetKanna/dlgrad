import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

cdef = """
void sum_3d(float *x, float *out, int *xshape, int *xstride, int outnumel, int dim);
void sum_2d(float *x, float *out, int *xshape, int *xstride, int outnumel, int dim);
void sum(float *x, float *out, int numel);
"""
ffi.cdef(cdef)

ffi.set_source("_sum", f"""
        #include "{root_dir}/src/c/sum.h"
    """,
    sources=[f'{root_dir}/src/c/sum.c'],
    extra_compile_args=["-O2", "-march=native"]
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
