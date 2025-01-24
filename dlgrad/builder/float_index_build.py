import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("void indexing(float *x, float *out, int *xshape, int *xstride, float *idxs);")
ffi.set_source("_index", f"""
    #include "{root_dir}/src/c/index.h"
""",
sources=[f'{root_dir}/src/c/index.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
