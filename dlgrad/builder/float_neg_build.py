import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("void neg(float *x, float *out, int numel);")
ffi.set_source("_neg", f"""
    #include "{root_dir}/src/c/neg.h"
""",
sources=[f'{root_dir}/src/c/neg.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
