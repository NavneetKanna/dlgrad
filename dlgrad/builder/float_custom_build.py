import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("void ce_backward(float *x, float *target, int *xshape, int *xstride);")
ffi.set_source("_custom", f"""
    #include "{root_dir}/src/c/custom.h"
""",
sources=[f'{root_dir}/src/c/custom.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
