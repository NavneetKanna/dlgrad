from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_allocate",
    headers=["allocate.h"],
    sources=["allocate.c"],
    cdef="""
        float *uninitialized_memory(size_t nbytes);
        float *initialized_memory(size_t num, size_t size);
        void free_ptr(float *ptr);
        float *init_with_scalar(size_t nbytes, int numel, int scalar);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
