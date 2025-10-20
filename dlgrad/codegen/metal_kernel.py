# ruff: noqa
from collections.abc import Generator
from typing import Any
from functools import cache


def n_gen() -> Generator[str, Any, None]:
    a = ["batch", "channel", "row", "col"][::-1]
    yield from a

@cache
def arithmetic(x_shape: tuple, x_stride: tuple, y_shape: tuple, y_stride: tuple, op: str):
    gen = n_gen()

    metal_code = """
        kernel void binary_op(
            const device float* x  [[ buffer(0) ]],
            const device float* y  [[ buffer(1) ]],
            device float* out      [[ buffer(2) ]],
            uint tid                [[ thread_position_in_grid ]]) 
        {
    """
    if x_shape == y_shape:
        metal_code += "uint y_idx = tid;\n"
    else:
        var_str = []
        metal_code += "uint temp = tid;\n"
        for i in x_shape[::-1]:
            var = next(gen)
            var_str.append(var)
            metal_code += f"uint {var} = temp % {i}; temp = temp / {i};\n"
        
        t = "uint y_idx = "
        for i, j, k in zip(y_stride, var_str[::-1], y_shape):
            if k != 1:
                t += f"{j}*{i} + "

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

@cache
def matmul(x_shape: tuple, y_shape: tuple):
    metal_code = f"""
    kernel void matmul(
            const device float* x  [[ buffer(0) ]],
            const device float* y  [[ buffer(1) ]],
            device float* out      [[ buffer(2) ]],
            uint2 tid              [[ thread_position_in_grid ]],
            uint2 gid              [[ threadgroup_position_in_grid ]],
            uint2 lid              [[ thread_position_in_threadgroup ]])
        {{
            uint row = tid.y;
            uint col = tid.x;

            float tmp = 0.0;
            for (int k=0; k<{x_shape[1]}; k++) {{
                tmp += x[row * {x_shape[1]} + k] * y[k*{y_shape[1]} + col];
            }}

            out[row * {x_shape[0]} + col] = tmp;
        }}
    """

    return metal_code

@cache
def max(x_shape: tuple, x_stride: tuple, dim: int, x_numel: int, out_stride: tuple, along_batch: bool = True):
    if along_batch:
        metal_code = f"""
            #include <metal_math>
            #include <metal_stdlib>
            using namespace metal;
            kernel void max(
                const device float* x  [[ buffer(0) ]],
                device float* out      [[ buffer(1) ]],
                uint2 tid              [[ thread_position_in_grid ]],
                uint2 gid              [[ threadgroup_position_in_grid ]],
                uint2 lid              [[ thread_position_in_threadgroup ]],
                uint2 threadgroup_size [[ threads_per_threadgroup ]],
                uint simd_size         [[ threads_per_simdgroup ]],
                uint simd_lane_id      [[ thread_index_in_simdgroup ]],
                uint simd_group_id     [[ simdgroup_index_in_threadgroup ]])
            {{
                uint row = tid.y;
                uint col = tid.x;
                uint x_idx = row*{x_stride[-2]} + col;
                float max_val = -FLT_MAX;
                for(uint i=0; i<{x_shape[dim]}; i++) {{
                    max_val = fmax(x[x_idx], max_val);
                    x_idx += {x_stride[dim]};
                }}

                out[row*{out_stride[-2]} + col] = max_val;
            }}
        """

        return metal_code