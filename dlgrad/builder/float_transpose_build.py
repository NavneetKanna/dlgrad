from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_transpose",
    headers=["transpose.h"],
    sources=["transpose.c"],
    cdef="""
        void transpose(float *x, float *out, int xrows, int xcols, int *xstride, int *outstride);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
