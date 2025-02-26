from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_sum",
    headers=["sum.h"],
    sources=["sum.c"],
    cdef="""
        void sum_3d(float *x, float *out, int *xshape, int *xstride, int outnumel, int dim);
        void sum_2d(float *x, float *out, int *xshape, int *xstride, int outnumel, int dim);
        void sum(float *x, float *out, int numel);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
