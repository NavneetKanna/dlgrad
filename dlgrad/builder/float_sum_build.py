import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("float *sum(float *x, int numel); void free_sum(float *ptr);")
ffi.set_source("_sum", f"""
    #include "{root_dir}/src/c/sum.h"
""", 
sources=[f'{root_dir}/src/c/sum.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)