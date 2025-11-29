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
def matmul(x_shape: tuple, y_shape: tuple):
    metal_code = f"""
    kernel void matmul(
            const device float* x  [[ buffer(0) ]],
            const device float* y  [[ buffer(1) ]],
            device float* out      [[ buffer(2) ]],
            uint2 tid              [[ thread_position_in_grid ]])
        {{
            int row = tid.y;
            int col = tid.x;

            if (row >= {x_shape[0]} || col >= {y_shape[1]}) return;

            float tmp = 0.0;
            for (int k=0; k<{x_shape[1]}; k++) {{
                tmp += x[row * {x_shape[1]} + k] * y[k*{y_shape[1]} + col];
            }}

            out[row * {y_shape[1]} + col] = tmp;
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
def transpose(x_shape: tuple):
    metal_code1 = f"""
        #include <metal_stdlib>
        using namespace metal;

        kernel void transpose(
            device const float* x    [[ buffer(0) ]],
            device float* out        [[ buffer(1) ]],
            uint2 tid                [[ thread_position_in_grid ]],
            uint2 tw                 [[ dispatch_threads_per_threadgroup ]],
            uint2 lid                [[ thread_position_in_threadgroup ]])
        {{
            int row = tid.y;
            int col = tid.x;

            threadgroup float ldata[32][32];

            int idx = row*{x_shape[1]} + col;
            ldata[lid.y][lid.x] = x[idx];

            threadgroup_barrier(mem_flags::mem_threadgroup);

            out[col*{x_shape[0]} + row] = ldata[lid.y][lid.x];
        }}
    """
    metal_code = f"""
    #include <metal_stdlib>
    using namespace metal;

    // Define tile size matching the threadgroup size
    #define TILE_DIM 32
    #define BLOCK_ROWS 32

    kernel void transpose(
        device const float* in     [[ buffer(0) ]],
        device float* out          [[ buffer(1) ]],
        // We need group ID and local ID to manually calculate swapped coordinates
        uint2 gid  [[ threadgroup_position_in_grid ]],
        uint2 tid  [[ thread_position_in_threadgroup ]])
    {{
        // Shared memory with padding to avoid bank conflicts
        // [32][33] instead of [32][32] offsets the stride by 1
        threadgroup float tile[TILE_DIM][TILE_DIM + 1];

        uint width = {x_shape[1]};
        uint height = {x_shape[0]};

        // 1. Coalesced Read
        // Calculate input global coordinates normally
        uint in_col = gid.x * TILE_DIM + tid.x;
        uint in_row = gid.y * TILE_DIM + tid.y;

        if (in_col < width && in_row < height) {{
            tile[tid.y][tid.x] = in[in_row * width + in_col];
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Transpose Block Coordinates
        // The block (x,y) in input becomes block (y,x) in output
        // We re-map tid.x to be the fastest moving dimension of the OUTPUT (new columns)
        uint out_col = gid.y * TILE_DIM + tid.x;
        uint out_row = gid.x * TILE_DIM + tid.y;

        // 3. Coalesced Write
        // We want to write to out[y_out * height + x_out].
        // Since x_out depends on tid.x, this write is coalesced.
        // But we need to grab the correct value from the tile.
        // The thread that is now handling (x_out, y_out) corresponds to
        // the transposed position in the tile: tile[tid.x][tid.y].
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
