import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("float *full(int numel, float fill_value); void *free_full(float *ptr);")
ffi.set_source("_full", f"""
    #include "{root_dir}/src/c/full.h"
""",
sources=[f'{root_dir}/src/c/full.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
