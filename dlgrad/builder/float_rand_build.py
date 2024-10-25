from cffi import FFI
from dlgrad.helpers import root_dir


ffi = FFI()

ffi.cdef("float *uni(int numel); void free_uni(float* ptr);")
ffi.set_source("_uni", f"""
    #include "{root_dir}/src/c/float_rand.h"
    #include "{root_dir}/src/c/pcg_basic.h"
""", 
sources=[f'{root_dir}/src/c/float_rand.c', f"{root_dir}/src/c/pcg_basic.c"],
libraries=["m"],
extra_compile_args=["-O3"])

if __name__ == "__main__":
    ffi.compile()