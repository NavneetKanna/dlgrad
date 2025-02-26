from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_max",
    headers=["max.h"],
    sources=["max.c"],
    cdef="""
        void max_3d(float *x, float *out, float *tmp, float *maxs_with_1s, int *xshape, int *xstride, int outnumel, int dim);
        void max_2d(float *x, float *out, float *tmp, float *maxs_with_1s, int *xshape, int *xstride, int outnumel, int dim);
        void max(float *x, float *out, int numel);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
