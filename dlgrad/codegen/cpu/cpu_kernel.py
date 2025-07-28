from collections.abc import Generator
from functools import cache
from typing import Any


def n_gen() -> Generator[str, Any, None]:
    a = ["i", "j", "k", "l"]
    yield from a


@cache
# def arithmetic(x: Buffer, y: Buffer, op: str) -> str:
def arithmetic(x_shape: tuple, x_stride: tuple, y_shape: tuple, y_stride: tuple, op: str) -> str:
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
