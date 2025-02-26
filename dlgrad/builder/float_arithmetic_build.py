from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_arithmetic",
    headers=["arithmetic.h"],
    sources=["arithmetic.c"],
    cdef="""
        void op_3d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op);
        void op_2d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op);
        void with_scalar(float *x, float *out, float *y, int xnumel, int op);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
