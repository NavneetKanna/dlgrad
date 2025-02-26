from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_mnist_loader",
    headers=["mnist.h"],
    sources=["mnist.c"],
    cdef="""
        float *mnist_images_loader(char *path, uint32_t magic_number);
        float *mnist_labels_loader(char *path, uint32_t magic_number);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
