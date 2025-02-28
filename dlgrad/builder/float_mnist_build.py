import os

from cffi import FFI

root_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))

ffi = FFI()

cdef = """
float *mnist_images_loader(char *path, uint32_t magic_number);
float *mnist_labels_loader(char *path, uint32_t magic_number);
"""
ffi.cdef(cdef)

ffi.set_source("_mnist_loader", f"""
        #include "{root_dir}/src/c/mnist.h"
    """,
    sources=[f'{root_dir}/src/c/mnist.c'],
    extra_compile_args=["-O2", "-march=native"]
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
