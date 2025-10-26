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

# Wasted a lot of time in trying to come up with a generic algo for all ndim, hence writing a separate one for each
@cache
def max_4d(x_shape: tuple, x_stride: tuple, dim: int):
    gen = n_gen()
    metal_code = """
        #include <metal_math>
        #include <metal_stdlib>
        using namespace metal;
        kernel void max(
            const device float* x  [[ buffer(0) ]],
            device float* out      [[ buffer(1) ]],
            uint2 tid              [[ thread_position_in_grid ]])
        {
            uint out_row = tid.y;
            uint out_col = tid.x;\n
    """
    if dim == 0:
        m = f"""
            uint x_idx = out_row*{x_shape[-1]} + out_col;
            float max_val = -FLT_MAX;
            for(uint i=0; i<{x_shape[dim]}; i++) {{
                max_val = fmax(x[x_idx], max_val);
                x_idx += {x_stride[-2]};
            }}
            out[out_row*{x_shape[-1]} + out_col] = max_val;
        }}
        """
        metal_code += m
        return metal_code
    elif dim == 1:
        metal_code += f"""
            uint batch = out_row / {x_shape[2]};
            uint row = out_row % {x_shape[2]};
            uint col = out_col;
            float max_val = -FLT_MAX;
            for (uint channel=0; channel<{x_shape[dim]}; channel++) {{\n
        """

        var_str = []
        t = "uint x_idx = "
        for i in x_stride[::-1]:
            var = next(gen)
            var_str.append(var)
            t += f"{i}*{var} + "
        t = t[:-3]
        t += ";\n"
        metal_code += t

        t = f"""
                max_val = fmax(x[x_idx], max_val); 
            }}
            out[out_row*{x_shape[-1]} + out_col] = max_val;
            }}
        """
        metal_code += t
        return metal_code
    elif dim == 2:
        metal_code += f"""
            float max_val = -FLT_MAX;
            uint x_idx = out_row*{x_stride[-3]} + out_col;
            for (uint row=0; row<{x_shape[dim]}; row++) {{\n
                max_val = fmax(x[x_idx], max_val);
                x_idx += {x_stride[dim]};
            }}
            out[out_row*{x_shape[-1]} + out_col] = max_val;
            }}
        """
        return metal_code
    elif dim == 3:
        metal_code = f"""
            #include <metal_math>
            #include <metal_stdlib>
            using namespace metal;
            kernel void max(
                const device float* x  [[ buffer(0) ]],
                device float* out      [[ buffer(1) ]],
                uint2 tid              [[ thread_position_in_grid ]],
                uint2 lid              [[ thread_position_in_threadgroup ]])
            {{
                uint row = tid.y;
                uint col = tid.x;

                threadgroup float tmp[32];
                for (uint i=0; i<32; i++) 
                    tmp[i] = -INFINITY;

                float local_max = -INFINITY;
                float v;
                for (uint i=col; i<{x_shape[3]}; i+=1024) {{
                    v = x[row*{x_shape[3]} + i];
                    local_max = max(local_max, v);
                }}

                for (int offset = 16; offset > 0; offset >>= 1)
                    local_max = max(local_max, simd_shuffle_down(local_max, offset));

                if (col % 32 == 0)
                    tmp[col / 32] = local_max;

                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (col < 32) {{
                    float max_val = tmp[col];
                    float val;
                    for (int offset = 16; offset > 0; offset >>= 1)
                        val = max(max_val, simd_shuffle_down(max_val, offset));
                    if (lid.x == 0)
                        out[row] = val;
                }}
            }}
        """
        return metal_code

@cache
def max_3d(x_shape: tuple, x_stride: tuple, dim: int):
    gen = n_gen()
    metal_code = """
        #include <metal_math>
        #include <metal_stdlib>
        using namespace metal;
        kernel void max(
            const device float* x  [[ buffer(0) ]],
            device float* out      [[ buffer(1) ]],
            uint2 tid              [[ thread_position_in_grid ]])
        {
            uint out_row = tid.y;
            uint out_col = tid.x;\n
    """

    if dim == 0:
        metal_code += f"""
            uint batch = out_row / {x_shape[1]};
            uint row = out_row % {x_shape[1]};
            uint col = out_col;
            float max_val = -FLT_MAX;
            for (uint channel=0; channel<{x_shape[dim]}; channel++) {{\n
        """

        var_str = []
        t = "uint x_idx = "
        for i in x_stride[::-1]:
            var = next(gen)
            var_str.append(var)
            t += f"{i}*{var} + "
        t = t[:-3]
        t += ";\n"
        metal_code += t

        t = f"""
                max_val = fmax(x[x_idx], max_val); 
            }}
            out[out_row*{x_shape[-1]} + out_col] = max_val;
            }}
        """
        metal_code += t
        return metal_code
    elif dim == 1:
        metal_code += f"""
            float max_val = -FLT_MAX;
            uint x_idx = out_row*{x_stride[-3]} + out_col;
            for (uint row=0; row<{x_shape[dim]}; row++) {{\n
                max_val = fmax(x[x_idx], max_val);
                x_idx += {x_stride[dim]};
            }}
            out[out_row*{x_shape[-1]} + out_col] = max_val;
            }}
        """
        return metal_code

