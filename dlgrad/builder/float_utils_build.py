from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_utils",
    headers=["utils.h"],
    sources=["utils.c"],
    cdef="""
        void neg(float *x, float *out, int numel);
        void cexp(float *x, float *out, int numel);
        void clog(float *x, float *out, int numel);
        void cpow(float *x, float *out, float val, int numel);
        void csqrt(float *x, float *out, int numel);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
