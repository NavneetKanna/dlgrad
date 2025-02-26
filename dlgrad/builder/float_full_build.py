from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_full",
    headers=["full.h"],
    sources=["full.c"],
    cdef="""
        void full(float *out, int numel, float fill_value);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
