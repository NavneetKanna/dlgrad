# ruff: noqa
from collections.abc import Generator
from typing import Any
from functools import cache
import math
from dlgrad.dtype import Scalar
import math


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
        metal_code += "int y_idx = tid;\n"
    else:
        var_str = []
        metal_code += "int temp = tid;\n"
        for i in x_shape[::-1]:
            var = next(gen)
            var_str.append(var)
            metal_code += f"int {var} = temp % {i}; temp = temp / {i};\n"

        t = "int y_idx = "
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
def matmul_3d(B, M, K, N):
    metal_code = f"""
        #include <metal_stdlib>
        using namespace metal;

        #define TILE_DIM 32

        kernel void matmul(
            const device float* x  [[ buffer(0) ]],
            const device float* y  [[ buffer(1) ]],
            device float* out      [[ buffer(2) ]],
            uint3 gid              [[ threadgroup_position_in_grid ]],
            uint3 tid              [[ thread_position_in_threadgroup ]])
        {{
            uint batch_idx = gid.z;
            if (batch_idx >= {B})
                return;

            int A_offset = batch_idx * {M} * {K};
            int B_offset = batch_idx * {K} * {N};
            int C_offset = batch_idx * {M} * {N};

            int out_row = gid.y * TILE_DIM + tid.y; // starting of the block
            int out_col = gid.x * TILE_DIM + tid.x; // starting of the block

            threadgroup float tileA[TILE_DIM][TILE_DIM];
            threadgroup float tileB[TILE_DIM][TILE_DIM];

            float sum = 0.0f;

            for (int k=0; k<{K}; k+=TILE_DIM) {{
                if (out_row < {M} && (k + tid.x) < {K}) {{
                    tileA[tid.y][tid.x] = x[A_offset + out_row * {K} + (k + tid.x)];
                }} else {{
                    tileA[tid.y][tid.x] = 0.0;
                }}

                if (out_col < {N} && (k+tid.y) < {K}) {{
                    tileB[tid.y][tid.x] = y[B_offset + (k + tid.y) * {N} + out_col];
                }} else {{
                    tileB[tid.y][tid.x] = 0.0;
                }}

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int i=0; i<TILE_DIM; i++) {{
                    sum += tileA[tid.y][i] * tileB[i][tid.x];
                }}

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            if (out_row < {M} && out_col < {N}) {{
                out[C_offset + out_row * {N} + out_col] = sum;
            }}
        }}
    """
    return metal_code

@cache
def matmul(x_shape: tuple, y_shape: tuple):
    metal_code = f"""
        #include <metal_stdlib>
        using namespace metal;

        #define TILE_DIM 32

        kernel void matmul(
            const device float* x  [[ buffer(0) ]],
            const device float* y  [[ buffer(1) ]],
            device float* out      [[ buffer(2) ]],
            uint2 gid              [[ threadgroup_position_in_grid ]],
            uint2 tid              [[ thread_position_in_threadgroup ]])
        {{
            int out_row = gid.y * TILE_DIM + tid.y; // starting of the block
            int out_col = gid.x * TILE_DIM + tid.x; // starting of the block

            threadgroup float tileA[TILE_DIM][TILE_DIM];
            threadgroup float tileB[TILE_DIM][TILE_DIM];

            float sum = 0.0f;

            for (int k=0; k<{x_shape[1]}; k+=TILE_DIM) {{
                if (out_row < {x_shape[0]} && (k + tid.x) < {x_shape[1]}) {{
                    tileA[tid.y][tid.x] = x[(out_row * {x_shape[1]}) + (k + tid.x)];
                }} else {{
                    tileA[tid.y][tid.x] = 0.0;
                }}

                if (out_col < {y_shape[1]} && (k+tid.y) < {y_shape[0]}) {{
                    tileB[tid.y][tid.x] = y[((k + tid.y) * {y_shape[1]}) + out_col];
                }} else {{
                    tileB[tid.y][tid.x] = 0.0;
                }}

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int i=0; i<TILE_DIM; i++) {{
                    sum += tileA[tid.y][i] * tileB[i][tid.x];
                }}

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            if (out_row < {x_shape[0]} && out_col < {y_shape[1]}) {{
                out[out_row * {y_shape[1]} + out_col] = sum;
            }}
        }}
    """
    return metal_code

@cache
def matmul_fast(x_shape: tuple, y_shape: tuple):
    metal_code = f"""
        #include <metal_simdgroup_matrix>
        using namespace metal;
        kernel void matmul(
            const device float* x  [[ buffer(0) ]],
            const device float* y  [[ buffer(1) ]],
            device float* out      [[ buffer(2) ]],
            uint2 gid              [[ threadgroup_position_in_grid ]])
        {{
            simdgroup_float8x8 matA, matB, matC(0.0f);

            int x_idx = gid.y * {x_shape[1]} * 8;
            const device float* x_ptr = x + x_idx;

            int y_idx = (gid.x * 8);
            const device float* y_ptr = y + y_idx;

            int out_idx = (gid.x * 8) + (gid.y * {y_shape[1]} * 8);
            device float* out_ptr = out + out_idx;

            for (int k=0; k<{x_shape[1]}; k+=8) {{
                simdgroup_load(matA, x_ptr + k, {x_shape[1]});
                simdgroup_load(matB, y_ptr + (k * {y_shape[1]}), {y_shape[1]});
                simdgroup_multiply_accumulate(matC, matA, matB, matC);
            }}

            simdgroup_store(matC, out_ptr, {y_shape[1]});
        }}
    """
    return metal_code

@cache
def reduce_along_rows(x_shape: tuple, op: str) -> str:
    if op == "max":
        t = "float val = -INFINITY;"
    elif op == "sum":
        t = "float val = 0.0;"

    metal_code = f"""
        #include <metal_math>
        #include <metal_stdlib>
        using namespace metal;
        kernel void {op}(
            const device float* x  [[ buffer(0) ]],
            device float* out      [[ buffer(1) ]],
            uint2 tid              [[ thread_position_in_grid ]],
            uint2 lid              [[ thread_position_in_threadgroup ]],
            uint simd_lane_id      [[ thread_index_in_simdgroup ]],
            uint simd_group_id     [[ simdgroup_index_in_threadgroup ]])
        {{
            int row = tid.y;
            int col = tid.x;

            if (col >= {x_shape[-1]}) return;
    """

    if op == "max":
        t = "tmp[lid.x] = -INFINITY;"
    elif op == "sum":
        t = "tmp[lid.x] = 0.0;"

    metal_code += f"""
            const int PARTIALS = {math.ceil(x_shape[-1]/32)};

            threadgroup float tmp[PARTIALS];
            if (lid.x < PARTIALS) {{
                {t}
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
        """

    if op == "max":
        t = "float val = -FLT_MAX;"
    elif op == "sum":
        t = "float val = 0.0;"

    metal_code += t

    if op == "max":
        opt = "val = fmax(val, v);"
    elif op == "sum":
        opt = "val += v;"

    metal_code += f"""
            float v;
            for (int i=col; i<{x_shape[-1]}; i+=1024) {{
                v = x[row*{x_shape[-1]} + i];
                {opt}
            }}
    """
    if op == "max":
        opt = "val = simd_max(val);"
    elif op == "sum":
        opt = "val = simd_sum(val);"

    metal_code += f"""
            {opt}
            if (simd_lane_id == 0)
                tmp[simd_group_id] = val;

            threadgroup_barrier(mem_flags::mem_threadgroup);
    """

    if op == "max":
        opt = "float val = (simd_lane_id < PARTIALS) ? tmp[simd_lane_id] : -INFINITY;"
    elif op == "sum":
        opt = "float val = (simd_lane_id < PARTIALS) ? tmp[simd_lane_id] : 0.0;"

    metal_code += f"""
            if (simd_group_id == 0) {{
                {opt}
    """

    if op == "max":
        opt = "val = simd_max(val);"
    elif op == "sum":
        opt = "val  = simd_sum(val);"

    metal_code += f"""
                {opt}
                if (simd_lane_id == 0)
                    out[row] = val;
            }}
        }}
    """
    return metal_code

@cache
def where(x_shape: tuple, inp: bool, other: bool):
    metal_code = f"""
        #include <metal_math>
        #include <metal_stdlib>
        using namespace metal;
        kernel void where(
            const device float* x  [[ buffer(0) ]],
            device float* out      [[ buffer(1) ]],
            device float* inp      [[ buffer(2) ]],
            device float* other    [[ buffer(3) ]],
            uint2 tid              [[ thread_position_in_grid ]])
        {{
            int row = tid.y;
            int col = tid.x;\n

            int idx = row*{x_shape[-1]} + col;
            if (x[idx] > 0.0)
    """

    if inp:
        ss = "out[idx] = inp[0];"
    else:
        ss = "out[idx] = inp[idx];"

    metal_code += ss

    metal_code += "\nelse\n"

    if other:
        ss = "out[idx] = other[0];"
    else:
        ss = "out[idx] = other[idx];"

    metal_code += ss

    metal_code += "}"

    return metal_code

@cache
def transpose_3d_01(x_shape: tuple, x_stride: tuple):
    metal_code = f"""
        #include <metal_stdlib>
        using namespace metal;

        kernel void transpose(
            device const float* x    [[ buffer(0) ]],
            device float* out        [[ buffer(1) ]],
            uint2 tid                [[ thread_position_in_grid ]])
        {{
            int base_x_idx = (tid.y / {x_shape[0]}) * {x_stride[1]};
            int x_idx = base_x_idx + ((tid.y % {x_shape[0]}) * {x_stride[0]});

            out[tid.y * {x_shape[2]} + tid.x] = x[x_idx + tid.x];
        }}
    """
    return metal_code

@cache
def transpose_3d_12(x_shape: tuple):
    metal_code = f"""
        #include <metal_stdlib>
        using namespace metal;

        #define TILE_DIM 32

        kernel void transpose(
            device const float* in     [[ buffer(0) ]],
            device float* out          [[ buffer(1) ]],
            uint3 gid                  [[ threadgroup_position_in_grid ]],
            uint3 tid                  [[ thread_position_in_threadgroup ]])
        {{
            threadgroup float tile[32][32 + 1];

            uint batch_size = {x_shape[0]};
            uint height = {x_shape[1]};
            uint width  = {x_shape[2]};

            uint batch_idx = gid.z;

            if (batch_idx >= batch_size)
                return;

            uint batch_offset = batch_idx * width * height;

            uint in_col = gid.x * TILE_DIM + tid.x;
            uint in_row = gid.y * TILE_DIM + tid.y;

            if (in_col < width && in_row < height) {{
                tile[tid.y][tid.x] = in[batch_offset + in_row * width + in_col];
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint out_col = gid.y * TILE_DIM + tid.x;
            uint out_row = gid.x * TILE_DIM + tid.y;

            if (out_row < width && out_col < height) {{
                out[batch_offset + out_row * height + out_col] = tile[tid.x][tid.y];
            }}
        }}
    """
    return metal_code

@cache
def transpose_2d(x_shape: tuple):
    metal_code = f"""
        #include <metal_stdlib>
        using namespace metal;

        #define TILE_DIM 32

        kernel void transpose(
            device const float* in     [[ buffer(0) ]],
            device float* out          [[ buffer(1) ]],
            uint2 gid                  [[ threadgroup_position_in_grid ]],
            uint2 tid                  [[ thread_position_in_threadgroup ]])
        {{
            threadgroup float tile[TILE_DIM][TILE_DIM + 1];

            uint width = {x_shape[1]};
            uint height = {x_shape[0]};

            uint in_col = gid.x * TILE_DIM + tid.x;
            uint in_row = gid.y * TILE_DIM + tid.y;

            if (in_col < width && in_row < height) {{
                tile[tid.y][tid.x] = in[in_row * width + in_col];
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint out_col = gid.y * TILE_DIM + tid.x;
            uint out_row = gid.x * TILE_DIM + tid.y;

            if (out_row < height && out_col < width) {{
                out[out_row * height + out_col] = tile[tid.x][tid.y];
            }}
        }}
    """
    return metal_code

@cache
def utils(x_shape: tuple, func: str, val: Scalar = None):
    match func:
        case "neg":
            metal_code = f"""
                #include <metal_stdlib>
                using namespace metal;

                kernel void neg(
                    device const float* x    [[ buffer(0) ]],
                    device float* out        [[ buffer(1) ]],
                    uint2 tid                [[ thread_position_in_grid ]])
                {{
                    int row = tid.y;
                    int col = tid.x;

                    int idx = row*{x_shape[-1]} + col;
                    out[idx] = -x[idx];
                }}
            """
            return metal_code
        case "exp":
            metal_code = f"""
                #include <metal_stdlib>
                #include <metal_math>
                using namespace metal;

                kernel void exp(
                    device const float* x    [[ buffer(0) ]],
                    device float* out        [[ buffer(1) ]],
                    uint2 tid                [[ thread_position_in_grid ]])
                {{
                    int row = tid.y;
                    int col = tid.x;

                    int idx = row*{x_shape[-1]} + col;
                    out[idx] = exp(x[idx]);
                }}
            """
            return metal_code
        case "log":
            metal_code = f"""
                #include <metal_stdlib>
                #include <metal_math>
                using namespace metal;

                kernel void log(
                    device const float* x    [[ buffer(0) ]],
                    device float* out        [[ buffer(1) ]],
                    uint2 tid                [[ thread_position_in_grid ]])
                {{
                    int row = tid.y;
                    int col = tid.x;

                    int idx = row*{x_shape[-1]} + col;
                    out[idx] = log(x[idx]);
                }}
            """
            return metal_code
        case "pow":
            metal_code = f"""
                #include <metal_stdlib>
                #include <metal_math>
                using namespace metal;

                kernel void pow(
                    device const float* x    [[ buffer(0) ]],
                    device float* out        [[ buffer(1) ]],
                    uint2 tid                [[ thread_position_in_grid ]])
                {{
                    int row = tid.y;
                    int col = tid.x;

                    int idx = row*{x_shape[-1]} + col;
                    out[idx] = pow(x[idx], {val});
                }}
            """
            return metal_code
        case "sqrt":
            metal_code = f"""
                #include <metal_stdlib>
                #include <metal_math>
                using namespace metal;

                kernel void sqrt(
                    device const float* x    [[ buffer(0) ]],
                    device float* out        [[ buffer(1) ]],
                    uint2 tid                [[ thread_position_in_grid ]])
                {{
                    int row = tid.y;
                    int col = tid.x;

                    int idx = row*{x_shape[-1]} + col;
                    out[idx] = sqrt(x[idx]);
                }}
            """
            return metal_code
