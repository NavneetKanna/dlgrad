import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

cdef = """
void neg(float *x, float *out, int numel);
void cexp(float *x, float *out, int numel);
void clog(float *x, float *out, int numel);
void cpow(float *x, float *out, float val, int numel);
void csqrt(float *x, float *out, int numel);
"""
ffi.cdef(cdef)

ffi.set_source("_utils", f"""
        #include "{root_dir}/src/c/utils.h"
    """,
    sources=[f'{root_dir}/src/c/utils.c'],
    libraries=["m"],
    extra_compile_args=["-O2", "-march=native"]
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
