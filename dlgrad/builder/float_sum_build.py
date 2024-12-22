import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("float *sum(float *x, int numel); void free_sum(float *ptr); \
         float *sum_3d_dim0(float *arr, int numel, int *shape, int *strides); \
         float *sum_3d_dim1(float *arr, int numel, int *shape, int *strides);\
         float *sum_3d_dim2(float *arr, int numel, int *shape, int *strides);")
ffi.set_source("_sum", f"""
    #include "{root_dir}/src/c/sum.h"
""", 
sources=[f'{root_dir}/src/c/sum.c'],
libraries=["m"],
extra_compile_args=["-O3", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)