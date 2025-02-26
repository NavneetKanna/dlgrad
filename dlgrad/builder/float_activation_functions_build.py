from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_af",
    headers=["activation_functions.h"],
    sources=["activation_functions.c"],
    cdef="""
        void relu(float *arr, float *out, int numel);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
