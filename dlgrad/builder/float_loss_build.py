import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

cdef = """
void ce_forward(float *x, float *target, float *out, int nrows, int *xstride);
void ce_backward(float *x, float *target, int *xshape, int *xstride);
"""
ffi.cdef(cdef)
ffi.set_source("_loss", f"""
    #include "{root_dir}/src/c/loss.h"
""",
sources=[f'{root_dir}/src/c/loss.c'],
libraries=["m"],
extra_compile_args=["-O2", "-march=native"])

if __name__ == "__main__":
    ffi.compile(verbose=True)
