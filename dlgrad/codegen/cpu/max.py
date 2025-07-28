# ruff :noqa

from dlgrad.buffer import Buffer

# def max(x: Buffer, dim: int) -> str:
#     code = f"""
#     #include <stdlib.h>

#     void maxx(float *x, float *out) {{
#         int shape_dim = {x.shape[dim]};
#         int stride_dim = {x.stride[dim]};
#         int numel = {x.numel};

#         int out_start = 0;
#         for (int j = 0; j < numel; j += stride_dim) {{
#             if ((j % (stride_dim * shape_dim)) == 0) {{
#                 if (j != 0) {{
#                     out_start += stride_dim;
#                 }} else {{
#                     out_start = 0;
#                 }}
#                 // copy
#                 for (int i = 0; i < stride_dim; i++) {{
#                     out[out_start + i] = x[j + i];
#                 }}
#             }} else {{
#                 // max
#                 for (int i = 0; i < stride_dim; i++) {{
#                     float val = x[j + i];
#                     if (val > out[out_start + i]) {{
#                         out[out_start + i] = val;
#                     }}
#                 }}
#             }}
#         }}
#     }}
#     """
#     return code

# def sum(x: Buffer, dim: int) -> str:
#     code = f"""
#     #include <stdlib.h>

#     void summ(float *x, float *out) {{
#         int shape_dim = {x.shape[dim]};
#         int stride_dim = {x.stride[dim]};
#         int numel = {x.numel};

#         int out_start = 0;
#         for (int j = 0; j < numel; j += stride_dim) {{
#             if ((j % (stride_dim * shape_dim)) == 0) {{
#                 if (j != 0) {{
#                     out_start += stride_dim;
#                 }} else {{
#                     out_start = 0;
#                 }}
#                 // copy
#                 for (int i = 0; i < stride_dim; i++) {{
#                     out[out_start + i] = x[j + i];
#                 }}
#             }} else {{
#                 // max
#                 for (int i = 0; i < stride_dim; i++) {{
#                     float val = x[j + i];
#                     out[out_start + i] += val;
#                 }}
#             }}
#         }}
#     }}
#     """
#     return code

"""
dim = 1

(4, 3, 2)
(6, 2, 1)

[[[0.40329522 0.7129083 ]
  [0.2902877  0.9216182 ]
  [0.73256075 0.8372984 ]]

 [[0.59929603 0.36998346]
  [0.55582845 0.17390463]
  [0.05803654 0.39682135]]

 [[0.7792665  0.47508177]
  [0.20465787 0.48576224]
  [0.01099625 0.86618644]]

 [[0.15955642 0.7351426 ]
  [0.49416125 0.13069092]
  [0.07697563 0.8316998 ]]]

#include <stdlib.h>

void maxx(float *x, float *out) {
    int shape_dim = 3;
    int stride_dim = 2;
    int numel = 24;

    int out_start = 0;
    for (int j = 0; j < numel; j += stride_dim) {
        if ((j % (stride_dim * shape_dim)) == 0) {
            if (j != 0) {
                out_start += stride_dim;
            } else {
                out_start = 0;
            }
            // copy
            for (int i = 0; i < stride_dim; i++) {
                out[out_start + i] = x[j + i];
            }
        } else {
            // max
            for (int i = 0; i < stride_dim; i++) {
                float val = x[j + i];
                if (val > out[out_start + i]) {
                    out[out_start + i] = val;
                }
            }
        }
    }
}

at loop index 0,
out_start=0, j=0
copies [0.40329522 0.7129083] into out[0] and out[1]

at loop index 2,
out_start=0, j=2
find max b/w x[2] and out[0] and b/w x[3] and out[1], if x[2] or x[3] is > out[0] or out[1], then replace

at loop index 4,
out_start=0, j=4
find max b/w x[4] and out[0] and b/w x[5] and out[1], if x[4] or x[5] is > out[0] or out[1], then replace

at loop index 6,
out_start=2, j=6
copies [0.59929603 0.36998346] into out[2] and out[3]

at loop index 8,
out_start=2, j=8
find max b/w x[8] and out[2] and b/w x[9] and out[3], if x[8] or x[9] is > out[2] or out[3], then replace

and so on.

This applies to any shape and any dim. This algo also ensures that we are avoiding cache miss when trying to find
max or sum along a dim, by looping through the input and output array's sequentially.
"""
