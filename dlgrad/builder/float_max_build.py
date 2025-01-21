import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("void max_3d(float *x, float *out, int *xshape, int *xstride, int outnumel, int dim);\
         void max_2d(float *x, float *out, int *xshape, int *xstride, int outnumel, int dim);\
         void max(float *x, float *out, int numel);")
ffi.set_source("_max", f"""
    #include "{root_dir}/src/c/max.h"
""",
sources=[f'{root_dir}/src/c/max.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
