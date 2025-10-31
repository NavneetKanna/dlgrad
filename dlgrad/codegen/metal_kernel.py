# ruff: noqa
from collections.abc import Generator
from typing import Any
from functools import cache
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
            uint2 tid              [[ thread_position_in_grid ]],
            uint2 gid              [[ threadgroup_position_in_grid ]],
            uint2 lid              [[ thread_position_in_threadgroup ]])
        {{
            int row = tid.y;
            int col = tid.x;

            if (row >= {x_shape[0]} || col >= {y_shape[1]}) return;   // M = rows of out, N = cols of out

            float tmp = 0.0;
            for (int k=0; k<{x_shape[1]}; k++) {{
                tmp += x[row * {x_shape[1]} + k] * y[k*{y_shape[1]} + col];
            }}

            out[row * {y_shape[1]} + col] = tmp;
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

# Wasted a lot of time in trying to come up with a generic algo for all ndim, hence writing a separate one for each
@cache
def reduce_4d(x_shape: tuple, x_stride: tuple, dim: int, op: str):
    gen = n_gen()
    metal_code = f"""
        #include <metal_math>
        #include <metal_stdlib>
        using namespace metal;
        kernel void {op}(
            const device float* x  [[ buffer(0) ]],
            device float* out      [[ buffer(1) ]],
            uint2 tid              [[ thread_position_in_grid ]])
        {{
            int out_row = tid.y;
            int out_col = tid.x;\n
    """
    if op == "max":
        t = "float val = -FLT_MAX;"
    elif op == "sum":
        t = "float val = 0.0;"
    
    metal_code += t

    if op == "max":
        opt = "val = fmax(x[x_idx], val);"
    elif op == "sum":
        opt = "val += x[x_idx];"

    if dim == 0:
        m = f"""
            int x_idx = out_row*{x_shape[-1]} + out_col;
            
            for(int i=0; i<{x_shape[dim]}; i++) {{
                {opt}
                x_idx += {x_stride[dim]};
            }}
            out[out_row*{x_shape[-1]} + out_col] = val;
        }}
        """
        metal_code += m
        return metal_code
    elif dim == 1:
        metal_code += f"""
            int batch = out_row / {x_shape[2]};
            int row = out_row % {x_shape[2]};
            int col = out_col;
            for (int channel=0; channel<{x_shape[dim]}; channel++) {{\n
        """

        var_str = []
        t = "int x_idx = "
        for i in x_stride[::-1]:
            var = next(gen)
            var_str.append(var)
            t += f"{i}*{var} + "
        t = t[:-3]
        t += ";\n"
        metal_code += t

        t = f"""
                {opt}
            }}
            out[out_row*{x_shape[-1]} + out_col] = val;
            }}
        """
        metal_code += t
        return metal_code
    elif dim == 2:
        metal_code += f"""
            int x_idx = out_row*{x_stride[-3]} + out_col;
            for (int row=0; row<{x_shape[dim]}; row++) {{\n
                {opt}
                x_idx += {x_stride[dim]};
            }}
            out[out_row*{x_shape[-1]} + out_col] = val;
            }}
        """
        return metal_code
    elif dim == 3:
        return reduce_along_rows(x_shape, op)

@cache
def max_3d(x_shape: tuple, x_stride: tuple, dim: int, op: str):
    gen = n_gen()
    metal_code = f"""
        #include <metal_math>
        #include <metal_stdlib>
        using namespace metal;
        kernel void {op}(
            const device float* x  [[ buffer(0) ]],
            device float* out      [[ buffer(1) ]],
            uint2 tid              [[ thread_position_in_grid ]])
        {{
            int out_row = tid.y;
            int out_col = tid.x;\n
    """

    if op == "max":
        t = "float val = -FLT_MAX;"
    elif op == "sum":
        t = "float val = 0.0;"
    
    metal_code += t

    if op == "max":
        opt = "val = fmax(x[x_idx], val);"
    elif op == "sum":
        opt = "val += x[x_idx];"

    if dim == 0:
        metal_code += f"""
            int batch = out_row / {x_shape[1]};
            int row = out_row % {x_shape[1]};
            int col = out_col;
            for (int channel=0; channel<{x_shape[dim]}; channel++) {{\n
        """

        var_str = []
        t = "int x_idx = "
        for i in x_stride[::-1]:
            var = next(gen)
            var_str.append(var)
            t += f"{i}*{var} + "
        t = t[:-3]
        t += ";\n"
        metal_code += t

        t = f"""
                {opt}
            }}
            out[out_row*{x_shape[-1]} + out_col] = val;
            }}
        """
        metal_code += t
        return metal_code
    elif dim == 1:
        metal_code += f"""
            int x_idx = out_row*{x_stride[-3]} + out_col;
            for (int row=0; row<{x_shape[dim]}; row++) {{\n
                {opt}
                x_idx += {x_stride[dim]};
            }}
            out[out_row*{x_shape[-1]} + out_col] = val;
            }}
        """
        return metal_code
    elif dim == 2:
        return reduce_along_rows(x_shape, op)

@cache
def max_2d(x_shape: tuple, x_stride: tuple, dim: int, op: str):
    metal_code = f"""
        #include <metal_math>
        #include <metal_stdlib>
        using namespace metal;
        kernel void {op}(
            const device float* x  [[ buffer(0) ]],
            device float* out      [[ buffer(1) ]],
            uint2 tid              [[ thread_position_in_grid ]])
        {{
            int out_row = tid.y;
            int out_col = tid.x;\n
    """

    if op == "max":
        t = "float val = -FLT_MAX;"
    elif op == "sum":
        t = "float val = 0.0;"
    
    metal_code += t

    if op == "max":
        opt = "val = fmax(x[x_idx], val);"
    elif op == "sum":
        opt = "val += x[x_idx];"

    if dim == 0:
        metal_code += f"""
            int x_idx = out_row*{x_shape[-1]} + out_col;
            for (int row=0; row<{x_shape[dim]}; row++) {{\n
                {opt}
                x_idx += {x_stride[dim]};
            }}
            out[out_row*{x_shape[-1]} + out_col] = val;
            }}
        """
        return metal_code
    elif dim == 1:
        return reduce_along_rows(x_shape, op)
        
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
def transpose(x_shape: tuple):
    metal_code = f"""
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

            // each thread loads its respective element into shared memory
            int idx = row*{x_shape[-1]} + col;
            ldata[lid.y][lid.x] = x[idx];

            threadgroup_barrier(mem_flags::mem_threadgroup);

            out[col*{x_shape[0]} + row] = ldata[lid.y][lid.x];
            //out[idx] = ldata[lid.x][lid.y];
        }}
    """
    return metal_code
