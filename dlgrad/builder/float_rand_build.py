from cffi import FFI
import os 


root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

ffi.cdef("float *uniform(int numel); void free_uniform(float* ptr);")
ffi.set_source("_uni", f"""
    #include "{root_dir}/src/c/float_rand.h"
    #include "{root_dir}/src/c/pcg_basic.h"
""", 
sources=[f'{root_dir}/src/c/float_rand.c', f"{root_dir}/src/c/pcg_basic.c"],
libraries=["m"],
extra_compile_args=["-O3"])

if __name__ == "__main__":
    ffi.compile()