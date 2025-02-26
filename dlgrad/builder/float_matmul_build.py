from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_matmul",
    headers=["matmul.h"],
    sources=["matmul.c"],
    cdef="""
        void matmul(float *x, float *y, float *out, int x_rows, int y_cols, int y_rows, int *ystride, int *xstride);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
