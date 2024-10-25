from cffi import FFI

ffi = FFI()
ffi.cdef("float *uni(int numel); void free_uni(float* ptr);")
ffi.set_source("_uni", """
    #include "dlgrad/src/c/float_rand.h"
    #include "dlgrad/src/c/pcg_basic.h"
""", 
sources=['dlgrad/src/c/float_rand.c', "dlgrad/src/c/pcg_basic.c"],
libraries=["m"],
extra_compile_args=["-O3"])

if __name__ == "__main__":
    ffi.compile()