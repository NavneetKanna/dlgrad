from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_uniform",
    headers=["float_rand.h", "pcg_basic.h"],
    sources=["float_rand.c", "pcg_basic.c"],
    cdef="""
        int uniform(float *out, int numel, float low, float high);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
