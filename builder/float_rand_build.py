from cffi import FFI

ffi = FFI()
ffi.cdef("float *uni(int numel); void free_uni(float* ptr);")
ffi.set_source("_uni", """
    #include "src/c/float_rand.h"
    #include "src/c/pcg_basic.h"
""", 
sources=['src/c/float_rand.c', "src/c/pcg_basic.c"],
libraries=["m"],
extra_compile_args=["-O3"])

if __name__ == "__main__":
    ffi.compile()