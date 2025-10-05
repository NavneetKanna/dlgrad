# ruff: noqa
from collections.abc import Generator
from typing import Any
from functools import cache


def n_gen() -> Generator[str, Any, None]:
    a = ["batch", "channel", "row", "col"][::-1]
    yield from a

@cache
def generate_binary_op_kernel(x_shape: tuple, x_stride: tuple, y_shape: tuple, y_stride: tuple, op: str):
    gen = n_gen()

    metal_code = """
        kernel void binary_op(
            const device float* x  [[ buffer(0) ]],
            const device float* y  [[ buffer(1) ]],
            device float* out      [[ buffer(2) ]],
            uint tid                [[ thread_position_in_grid ]]) 
        {
    """
    var_str = []
    metal_code += "uint temp = tid;\n"
    t = ""
    for i in y_shape[::-1]:
        var = next(gen)
        var_str.append(var)
        if i == 1:
            continue
        t += f"uint {var} = temp % {i}; "
        t += f"temp = temp / {i};"
        t += "\n"
    
    last_semicolon = t.rfind(';')
    second_last_semicolon = t.rfind(';', 0, last_semicolon)
    t = t[:second_last_semicolon + 1]
    t += "\n"
    metal_code += t

    t = "uint y_idx = "
    for i, j, k in zip(y_stride, var_str[::-1], y_shape):
        if k == 1:
            continue
        else:
            t += f"{j} * {i} + "

    t = t[:-3]
    t += ";\n"
    metal_code += t

    match op:
        case "add":
            metal_code += "out[tid] = x[tid] + y[y_idx];\n"
        case "sub":
            metal_code += "out[tid] = x[tid] - y[y_idx];\n"
        case "mul":
            metal_code += "out[tid] = x[tid] * y[y_idx];\n"
        case "div":
            metal_code += "out[tid] = x[tid] / y[y_idx];\n"

    metal_code += "}\n"

    return metal_code
