from dlgrad.builder.builder_utils import build_extension

ffi = build_extension(
    module_name="_cmp",
    headers=["comparision.h"],
    sources=["comparision.c"],
    cdef="""
        void gt_with_scalar(float *arr, float *out, float val, int numel);
    """
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
