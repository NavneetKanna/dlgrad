from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_loss",
    headers=["loss.h"],
    sources=["loss.c"],
    cdef="""
        void ce_forward(float *x, float *target, float *out, int nrows, int *xstride);
        void ce_backward(float *x, float *target, int *xshape, int *xstride);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
