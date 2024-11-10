import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef(
    "float *add_2d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride); \
    float *add_3d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride, int yshape_len); \
        void free_add(float* ptr);")
ffi.set_source("_add", f"""
    #include "{root_dir}/src/c/add.h"
""", 
sources=[f'{root_dir}/src/c/add.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
