from collections.abc import Generator
from functools import cache
from typing import Any


def n_gen() -> Generator[str, Any, None]:
    a = ["i", "j", "k", "l"]
    yield from a


@cache
def arithmetic(x_shape: tuple, x_stride: tuple, y_shape: tuple, y_stride: tuple, op: str) -> tuple[str, str]:
    gen = n_gen()

    code  = f"""
    #include <stdlib.h>

    int x_off, y_off;
    void {op}(float *x, float *y, float *out) {{
    """

    var_str = []
    for i in x_shape:
        var = next(gen)
        var_str.append(var)
        code += f"""
        for (int {var}=0; {var}<{i}; {var}++) {{
        """

    ts = "x_off = "
    for i, v in zip(x_stride, var_str):
        ts += f"{i}*{v} + "
    ts = ts[:-3]
    ts += ";"

    code += ts

    code += "\n"

    ts = "y_off = "
    for idx, (i, j) in enumerate(zip(x_shape, y_shape)):
        if i==j and i != 1 and j != 1:
            ts += f"{y_stride[idx]}*{var_str[idx]} + "
    ts = ts[:-3]
    ts += ";"

    code += ts

    match op:
        case "add":
            code += """
                out[x_off] = x[x_off] + y[y_off];
            """
        case "sub":
            code += """
                out[x_off] = x[x_off] - y[y_off];
            """
        case "mul":
            code += """
                out[x_off] = x[x_off] * y[y_off];
            """
        case "div":
            code += """
                out[x_off] = x[x_off] / y[y_off];
            """

    for i in range(len(var_str)):
        code += "}\n"

    code += "}\n"

    return code, f"void {op}(float *x, float *y, float *out);"

@cache
def max(x_shape: tuple, x_stride: tuple, x_numel: int, dim: int) -> tuple[str, str]:
    if dim == -1:
        code = f"""
            void max(float *x, float *out) {{
                float m = 0.0;
                for (int i=0; i<{x_numel}; i++) {{
                    if (x[i] > m) {{
                        m = x[i];
                    }}
                }}
                out[0] = m;
            }}
        """

        return code, "void max(float *x, float *out);"

    code  = f"""
    void max(float *x, float *out) {{
        int shape_dim = {x_shape[dim]};
        int stride_dim = {x_stride[dim]};
        int numel = {x_numel};

        int out_start = 0;
        for (int j = 0; j < numel; j += stride_dim) {{
            if ((j % (stride_dim * shape_dim)) == 0) {{
                if (j != 0) {{
                    out_start += stride_dim;
                }} else {{
                    out_start = 0;
                }}
                // copy
                for (int i = 0; i < stride_dim; i++) {{
                    out[out_start + i] = x[j + i];
                }}
            }} else {{
                // max
                for (int i = 0; i < stride_dim; i++) {{
                    float val = x[j + i];
                    if (val > out[out_start + i]) {{
                        out[out_start + i] = val;
                    }}
                }}
            }}
        }}
    }}
    """

    return code, "void max(float *x, float *out);"

@cache
def sum(x_shape: tuple, x_stride: tuple, x_numel: int, dim: int) -> tuple[str, str]:
    if dim == -1:
        code = f"""
            void sum(float *x, float *out) {{
                float s = 0.0;
                for (int i=0; i<{x_numel}; i++) {{
                    s += x[i];
                }}
                out[0] = s;
            }}
        """

        return code, "void sum(float *x, float *out);"

    code  = f"""
    void sum(float *x, float *out) {{
        int shape_dim = {x_shape[dim]};
        int stride_dim = {x_stride[dim]};
        int numel = {x_numel};

        int out_start = 0;
        for (int j = 0; j < numel; j += stride_dim) {{
            if ((j % (stride_dim * shape_dim)) == 0) {{
                if (j != 0) {{
                    out_start += stride_dim;
                }} else {{
                    out_start = 0;
                }}
                // copy
                for (int i = 0; i < stride_dim; i++) {{
                    out[out_start + i] = x[j + i];
                }}
            }} else {{
                // sum
                for (int i = 0; i < stride_dim; i++) {{
                    float val = x[j + i];
                    out[out_start + i] += val;
                }}
            }}
        }}
    }}
    """

    return code, "void sum(float *x, float *out);"

@cache
def transpose_given_axes(x_shape: tuple, axes: tuple) -> list:
    ax = list(range(len(x_shape)))
    # replace elements at positions in axes with elements of axes reversed
    for i, pos in enumerate(axes):
        ax[pos] = axes[-(i+1)]
    return ax

@cache
def transpose(x_shape: tuple, x_stride: tuple,  out_stride: tuple, x_numel: int, axes: tuple) -> tuple[str, str]:
    gen = n_gen()

    ax = transpose_given_axes(x_shape, axes)

    code = """
    void transpose(float *x, float *out) {
    int out_idx = 0;
    int x_idx = 0;
    """

    var_str = []
    for i in x_shape:
        var = next(gen)
        var_str.append(var)
        code += f"""
        for (int {var}=0; {var}<{i}; {var}++) {{
        """

    ts = "out_idx = "
    for i, v in zip(out_stride, ax):
        ts += f"{i}*{var_str[v]} + "
    ts = ts[:-3]
    ts += ";"

    code += ts

    code += "\n"

    ts = "x_idx = "
    for i, v in zip(x_stride, var_str):
        ts += f"{i}*{v} + "
    ts = ts[:-3]
    ts += ";"

    code += ts

    code += "\n"

    code += """
    out[out_idx] = x[x_idx];
    """

    for i in range(len(var_str)):
        code += "}\n"

    code += "}\n"

    return code, "void transpose(float *x, float *out);"


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
