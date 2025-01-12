import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("void relu(float *arr, float *out, int numel);")
ffi.set_source("_af", f"""
    #include "{root_dir}/src/c/activation_functions.h"
""",
sources=[f'{root_dir}/src/c/activation_functions.c'],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
