from cffi import FFI
import os 
from pathlib import Path


ffi = FFI()

dir_path = f"{Path(__file__).parent.parent}/src/c"

ffi.cdef("float *uni(int numel); void free_uni(float* ptr);")
ffi.set_source("_uni", f"""
    #include {dir_path}/float_rand.h"
    #include "{dir_path}/pcg_basic.h"
""", 
sources=[f'{dir_path}/float_rand.c', f"{dir_path}/pcg_basic.c"],
libraries=["m"],
extra_compile_args=["-O3"])

if __name__ == "__main__":
    ffi.compile()