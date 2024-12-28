import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("float *gt_with_scalar(float *arr, float val, int numel); void free_cmp(float *ptr);")
ffi.set_source("_cmp", f"""
    #include "{root_dir}/src/c/comparision.h"
""",
sources=[f'{root_dir}/src/c/comparision.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
