import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

cdef = """
float *uninitialized_memory(size_t nbytes);
float *initialized_memory(size_t num, size_t size);
void free_ptr(float *ptr);
float *init_with_scalar(size_t nbytes, int numel, int scalar);
"""
ffi.cdef(cdef)

ffi.set_source("_allocate", f"""
        #include "{root_dir}/src/c/allocate.h"
    """,
    sources=[f'{root_dir}/src/c/allocate.c'],
    extra_compile_args=["-O2", "-march=native"]
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
