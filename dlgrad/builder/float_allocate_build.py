import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef(
    "float *uninitialized_memory(size_t num); float *initialized_memory(size_t num); void free_ptr(float *ptr);")
ffi.set_source("_allocate", f"""
    #include "{root_dir}/src/c/allocate.h"
""",
sources=[f'{root_dir}/src/c/allocate.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
