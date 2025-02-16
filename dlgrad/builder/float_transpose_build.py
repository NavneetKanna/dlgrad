import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

cdef = """
void transpose(float *x, float *out, int xrows, int xcols, int *xstride, int *outstride);
"""
ffi.cdef(cdef)

ffi.set_source("_transpose", f"""
        #include "{root_dir}/src/c/transpose.h"
    """,
    sources=[f'{root_dir}/src/c/transpose.c'],
    extra_compile_args=["-O2", "-march=native"]
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
