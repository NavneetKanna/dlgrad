import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("int uniform(float *out, int numel, float low, float high);")

ffi.set_source("_uniform", f"""
        #include "{root_dir}/src/c/float_rand.h"
        #include "{root_dir}/src/c/pcg_basic.h"
    """,
    sources=[f'{root_dir}/src/c/float_rand.c', f"{root_dir}/src/c/pcg_basic.c"],
    extra_compile_args=["-O2", "-march=native"]
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
