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

# TODO: id dim is None
@cache
def max(x_shape: tuple, x_stride: tuple, x_numel: int, dim: int) -> tuple[str, str]:
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
