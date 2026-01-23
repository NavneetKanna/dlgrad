from collections.abc import Generator
from functools import cache
from typing import Any

from dlgrad.dtype import Scalar


# NOTE: Assumes max 4D tensor
def n_gen() -> Generator[str, Any, None]:
    a = ["i", "j", "k", "l"]
    yield from a

@cache
def arithmetic(op: str, ndim: int) -> tuple[str, str]:
    """
    Generates a kernel for the unary ops.
    """
    ops = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/',
        'eq': '==',
        'lt': '<',
        'gt': '>'
    }
    c_op = ops[op]

    # Generate the Function Signature
    func_name = f"{op}_{ndim}d"
    cdef = f"void {func_name}(float *x, float *y, float *out, int *shape, int *x_stride, int *y_stride);"

    # Generate Nested Loops
    # "for (int i0=0; i0<shape[0]; i0++) { ..."
    loops = ""
    indent = "    "

    for d in range(ndim):
        loops += f"{indent * (d + 1)}for (int i{d} = 0; i{d} < shape[{d}]; i{d}++) {{\n"

    # Generate Index Calculation
    # x_idx = i0*x_stride[0] + i1*x_stride[1] ...
    body_indent = indent * (ndim + 1)

    x_offset_parts = [f"i{d}*x_stride[{d}]" for d in range(ndim)]
    y_offset_parts = [f"i{d}*y_stride[{d}]" for d in range(ndim)]

    calc_x = " + ".join(x_offset_parts) if ndim > 0 else "0"
    calc_y = " + ".join(y_offset_parts) if ndim > 0 else "0"

    closing_braces = ""
    for d in range(ndim, 0, -1):
        closing_braces += f"{indent * d}}}\n"

    c_code = f"""
        void {func_name}(float *x, float *y, float *out, int *shape, int *x_stride, int *y_stride) {{
            int ptr = 0;
            {loops}
                {body_indent}int x_idx = {calc_x};
                {body_indent}int y_idx = {calc_y};

                {body_indent}out[ptr++] = x[x_idx] {c_op} y[y_idx];
            {closing_braces}
        }}
    """

    return c_code, cdef

@cache
def reduce_source(op: str, ndim: int, reduce_dim: int) -> tuple[str, str]:
    """
    Generates a kernel for reduction operations (sum, max).
    """
    # Setup Ops
    if op == "sum":
        init_val = "float val = 0;"
        update_op = "val += v;"
        final_op = "val"
    elif op == "max":
        init_val = "float val = -FLT_MAX;"
        update_op = "if (v > val) val = v;"
        final_op = "val"
    elif op == "mean":
        init_val = "float val = 0;"
        update_op = "val += v;"
        final_op = f"val / shape[{reduce_dim}]"
    else:
        raise ValueError(f"Unknown reduce op: {op}")

    func_name = f"reduce_{op}_{ndim}d_axis{reduce_dim}"
    cdef = f"void {func_name}(const float *x, float *out, int *shape, int *x_stride, int *out_stride);"

    # Identify Dimensions
    # all_dims: [0, 1, 2, 3]
    # outer_dims: The dimensions we keep (Loops outside the accumulator)
    all_dims = range(ndim)
    outer_dims = [d for d in all_dims if d != reduce_dim]

    # Pre-calculate Index Math strings
    # Input index uses ALL iterators (i0, i1...)
    x_idx_parts = [f"i{d}*x_stride[{d}]" for d in all_dims]
    calc_x_idx = " + ".join(x_idx_parts)

    # Output index uses ONLY outer iterators.
    # We map them to out_stride[0], out_stride[1]... sequentially.
    out_idx_parts = []
    for k, d in enumerate(outer_dims):
        out_idx_parts.append(f"i{d}*out_stride[{k}]")
    calc_out_idx = " + ".join(out_idx_parts) if out_idx_parts else "0"

    body = ""
    indent = "    "

    # Open Outer Loops
    for i, d in enumerate(outer_dims):
        body += f"{indent * (i + 1)}for (int i{d} = 0; i{d} < shape[{d}]; i{d}++) {{\n"

    depth = len(outer_dims) + 1

    # Initialize Accumulator
    body += f"{indent * depth}{init_val}\n"

    # Open Inner Loop (Reduction)
    body += f"{indent * depth}for (int i{reduce_dim} = 0; i{reduce_dim} < shape[{reduce_dim}]; i{reduce_dim}++) {{\n"

    # Inner Logic
    body += f"{indent * (depth + 1)}float v = x[{calc_x_idx}];\n"
    body += f"{indent * (depth + 1)}{update_op}\n"

    # Close Inner Loop
    body += f"{indent * depth}}}\n"

    # Write Back Result
    body += f"{indent * depth}out[{calc_out_idx}] = {final_op};\n"

    # Close Outer Loops (in reverse)
    for i in range(len(outer_dims) - 1, -1, -1):
        body += f"{indent * (i + 1)}}}\n"

    final_code = f"""
        #include <stdlib.h>
        #include <float.h>

        void {func_name}(const float *x, float *out, int *shape, int *x_stride, int *out_stride) {{
            {body}
        }}
    """

    return final_code, cdef

@cache
def reduce(op: str) -> tuple[str, str]:
    if op == "mean":
        init = "float val = 0;"
        update = "val += x[i];"
        finalize = "out[0] = val / numel;"
    elif op == "sum":
        init = "float val = 0;"
        update = "val += x[i];"
        finalize = "out[0] = val;"
    elif op == "max":
        init = "float val = -FLT_MAX;"
        update = "if (x[i] > val) val = x[i];"
        finalize = "out[0] = val;"

    func_name = f"reduce_{op}"
    c_code = f"""
    #include <float.h>
    void {func_name}(const float *x, float *out, int numel) {{
        {init}
        for (int i = 0; i < numel; i++) {{
            {update}
        }}
        {finalize}
    }}
    """
    cdef = f"void {func_name}(const float *x, float *out, int numel);"
    return c_code, cdef

@cache
def permute(ndim: int) -> tuple[str, str]:
    func_name = f"permute_{ndim}d"
    cdef = f"void {func_name}(const float *x, float *out, int *shape, int *in_strides, int *out_strides);"

    # Generate Nested Loops based on Output Shape
    loops = ""
    indent = "    "
    for d in range(ndim):
        loops += f"{indent * (d + 1)}for (int i{d} = 0; i{d} < shape[{d}]; i{d}++) {{\n"

    # Calculate Indices
    # The 'in_strides' passed to this function will be the PERMUTED strides
    body_indent = indent * (ndim + 1)

    in_offset_parts = [f"i{d}*in_strides[{d}]" for d in range(ndim)]
    calc_in_idx = " + ".join(in_offset_parts)

    out_offset_parts = [f"i{d}*out_strides[{d}]" for d in range(ndim)]
    calc_out_idx = " + ".join(out_offset_parts)

    # Close Loops
    closing_braces = ""
    for k in range(ndim - 1, -1, -1):
        depth = k + 1
        closing_braces += f"{indent * depth}}}\n"

    c_code = f"""
    void {func_name}(const float *x, float *out, int *shape, int *in_strides, int *out_strides) {{
        {loops}
            {body_indent}int in_idx = {calc_in_idx};
            {body_indent}int out_idx = {calc_out_idx};
            {body_indent}out[out_idx] = x[in_idx];
        {closing_braces}
    }}
    """
    return c_code, cdef

@cache
def embedding(idx_numel: int, x_width: int) -> tuple[str, str]:
    c_code = f"""
        #include <string.h>
        void embedding(float *x, float *idx, float *out) {{
            for (int i=0; i<{idx_numel}; i++) {{
                int x_idx = idx[i]*{x_width};
                memcpy(&out[i * {x_width}], &x[x_idx], {x_width} * sizeof(float));
            }}
        }}

    """
    return c_code, "void embedding(float *x, float *idx, float *out);"

@cache
def embedding_backward(idx_numel: int, x_width: int) -> tuple[str, str]:
    c_code = f"""
        void embedding_backward(float *out, float *upstream_grad, float *idx) {{
            for (int i=0; i<{idx_numel}; i++) {{
                for (int j=0; j<{x_width}; j++) {{
                    int x_idx = idx[i]*{x_width}+j;
                    out[x_idx] += upstream_grad[i*{x_width}+j];
                }}
            }}
        }}
    """
    return c_code, "void embedding_backward(float *out, float *upstream_grad, float *idx);"

@cache
def utils(x_numel: int, func: str, val: Scalar = None) -> tuple[str, str]:
    match func:
        case "neg":
            code = f"""
            #include <math.h>

            void c_neg(float *x, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = -1 * x[i];
                }}
            }}
            """
            return code, "void c_neg(float *x, float *out);"
        case "exp":
            code = f"""
            #include <math.h>

            void c_exp(float *x, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = exp(x[i]);
                }}
            }}
            """
            return code, "void c_exp(float *x, float *out);"
        case "log":
            code = f"""
            #include <math.h>

            void c_log(float *x, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = log(x[i]);
                }}
            }}
            """
            return code, "void c_log(float *x, float *out);"
        case "pow":
            code = f"""
            #include <math.h>

            void c_pow(float *x, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = powf(x[i], {val});
                }}
            }}
            """
            return code, "void c_pow(float *x, float *out);"
        case "sqrt":
            code = f"""
            #include <math.h>

            void c_sqrt(float *x, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = sqrtf(x[i]);
                }}
            }}
            """
            return code, "void c_sqrt(float *x, float *out);"
        case "rsqrt":
            code = f"""
            #include <math.h>

            void c_rsqrt(float *x, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = 1.0/sqrtf(x[i]);
                }}
            }}
            """
            return code, "void c_rsqrt(float *x, float *out);"
@cache
def clamp(x_numel: int, min_val: int, max_val: int) -> tuple[str, str]:
    c_code = f"""
    void clamp(float *x, float *out) {{
        for (int i=0; i<{x_numel}; i++) {{
    """
    if min_val is not None and max_val is not None:
        c_code += f"""
        if (x[i] < {min_val}) {{
            out[i] = {min_val};
        }} else if (x[i] > {max_val}) {{
            out[i] = {max_val};
        }} else {{
            out[i] = x[i];
        }}
        """
    elif min_val is not None:
        c_code += f"""
        if (x[i] < {min_val}) {{
            out[i] = {min_val};
        }} else {{
            out[i] = x[i];
        }}
        """
    elif max_val is not None:
        c_code += f"""
        if (x[i] > {max_val}) {{
            out[i] = {max_val};
        }} else {{
            out[i] = x[i];
        }}
        """
    else:
        c_code += """
        out[i] = x[i];
        """
    c_code += """
                }
            }
        """
    return c_code, "void clamp(float *x, float *out);"

@cache
def matmul_4d(dims: tuple, x_stride: tuple, y_stride: tuple, out_stride: tuple) -> tuple[str, str]:
    B, C, M, K, N = dims

    c_code = f"""
        #include <stdint.h>

        void matmul_4d(const float* x, const float* y, float* out)
        {{
            for (int b = 0; b < {B}; ++b) {{
                for (int c = 0; c < {C}; ++c) {{
                    const float* A_ptr = x + (b * {x_stride[0]}) + (c * {x_stride[1]});
                    const float* B_ptr = y + (b * {y_stride[0]}) + (c * {y_stride[1]});
                    float* out_ptr = out + (b * {out_stride[0]}) + (c * {out_stride[1]});

                    for (int h = 0; h < {M}; ++h) {{
                        const float* A_row = A_ptr + (h * {x_stride[2]});
                        float* out_row = out_ptr + (h * {out_stride[2]});

                        for (int k = 0; k < {K}; ++k) {{
                            float a_val = A_row[k * {x_stride[3]}];
                            const float* B_row_k = B_ptr + (k * {y_stride[2]});

                            for (int w = 0; w < {N}; ++w) {{
                                out_row[w * {out_stride[3]}] += a_val * B_row_k[w * {y_stride[3]}];
                            }}
                        }}
                    }}
                }}
            }}
        }}
    """
    return c_code, "void matmul_4d(const float* x, const float* y, float* out);"

@cache
def matmul_3d(x_shape: tuple, y_shape: tuple, x_stride: tuple, y_stride: tuple, out_stride: tuple, broadcast_x: bool = False, broadcast_y: bool = False) -> tuple[str, str]:
    B = x_shape[0] if not broadcast_x else y_shape[0]
    M = x_shape[1]
    K = x_shape[2]
    N = y_shape[2]

    tx = f"b*{x_stride[0]} + i*{x_stride[1]} + k*{x_stride[2]}"
    ty = f"b*{y_stride[0]} + k*{y_stride[1]} + j*{y_stride[2]}"

    if broadcast_x:
        tx = f"i*{x_stride[1]} + k*{x_stride[2]}"
    if broadcast_y:
        ty = f"k*{y_stride[1]} + j*{y_stride[2]}"

    c_code = f"""
    void matmul_3d(float *x, float *y, float *out) {{
    for (int b=0; b<{B}; b++) {{
        for (int i=0; i<{M}; i++) {{
            for (int k=0; k<{K}; k++) {{
                float a = x[{tx}];
                for (int j=0; j<{N}; j++) {{
                    out[b*{out_stride[0]} + i*{out_stride[1]} + j*{out_stride[2]}] += a * y[{ty}];
                        }}
                    }}
                }}
            }}
        }}
    """
    return c_code, "void matmul_3d(float *x, float *y, float *out);"

@cache
def matmul_2d(x_shape: tuple, y_shape: tuple, x_stride: tuple, y_stride: tuple) -> tuple[str, str]:
    c_code = f"""
    void matmul_2d(float *x, float *y, float *out) {{
        for (int i=0; i<{x_shape[0]}; i++) {{
            for (int k=0; k<{x_shape[1]}; k++) {{
                float a = x[i*{x_stride[0]} + k*{x_stride[1]}];
                for (int j=0; j<{y_shape[1]}; j++) {{
                    out[i*{y_shape[1]} + j] += a * y[k*{y_stride[0]} + j*{y_stride[1]}];
                }}
            }}
        }}
    }}
    """
    return c_code, "void matmul_2d(float *x, float *y, float *out);"

@cache
def ce_forward(n_rows: int, x_stride: tuple) -> tuple[str, str]:
    c_code = f"""
    void ce_forward(float *x, float *target, float *out)
    {{
        for (int i=0; i<{n_rows}; i++) {{
            out[i] = x[(int)target[i]+({x_stride[0]}*i)];
        }}
    }}
    """

    return c_code, "void ce_forward(float *x, float *target, float *out);"

@cache
def ce_backward(x_shape: tuple, x_stride: tuple) -> tuple[str, str]:
    c_code = f"""
    void ce_backward(float *x, float *target)
    {{
        int rows = {x_shape[0]};
        int cols = {x_shape[1]};

        for (int i=0; i<rows; i++) {{
            x[(int)target[i]+({x_stride[0]}*i)] -= 1;
        }}
    }}
    """

    return c_code, "void ce_backward(float *x, float *target);"

@cache
def argmax(x_shape: tuple, dim: int) -> tuple[str, str]:
    if dim == 1:
        code = f"""
            void argmax2d(float *x, float *out)
            {{
                int rows = {x_shape[0]};
                int cols = {x_shape[1]};
                if ({dim} == 1) {{
                    for (int i = 0; i < rows; i++) {{
                        float max = x[i * cols + 0];
                        int idx = 0;
                        for (int j = 1; j < cols; j++) {{
                            if (x[i * cols + j] > max) {{
                                max = x[i * cols + j];
                                idx = j;
                            }}
                        }}
                        out[i] = idx;
                    }}
                }}
            }}
        """
        return code, "void argmax2d(float *x, float *out);"
    elif dim == 0:
        f"""
            void argmax2d(float *x, float *out)
            {{
                int rows = {x_shape[0]};
                int cols = {x_shape[1]};
                for (int j = 0; j < cols; j++) {{
                    float max = x[0 * cols + j];
                    int idx = 0;
                    for (int i = 1; i < rows; i++) {{
                        if (x[i * cols + j] > max) {{
                            max = x[i * cols + j];
                            idx = i;
                        }}
                    }}
                    out[j] = idx;
                }}
            }}
        """
        return code, "void argmax2d(float *x, float *out);"
    else:
        f"""
            void argmax2d(float *x, float *out)
            {{
                int rows = {x_shape[0]};
                int cols = {x_shape[1]};
                float max = -999;
                int idx = 0;
                for (int i=0; i<rows*cols; i++) {{
                    if (x[i] > max) {{
                        max = x[i];
                        idx = i;
                    }}
                }}
                out[0] = idx;
            }}
        """
        return code, "void argmax2d(float *x, float *out);"

@cache
def relu(x_numel: int) -> tuple[str, str]:
    code = f"""
        void relu(float *arr, float *out) {{
            for (int i=0; i<{x_numel}; i++) {{
                if (arr[i] <= 0) {{
                    out[i] = 0.0;
                }} else {{
                    out[i] = arr[i];
                }}
            }}
        }}
    """

    return code, "void relu(float *arr, float *out);"

@cache
def gt(x_numel: int, val: int | float) -> tuple[str, str]:
    code = f"""
        void gt_with_scalar(float *arr, float *out)
        {{
            for (int i=0; i<{x_numel}; i++) {{
                if (arr[i] > {val})
                    out[i] = 1.0;
                else
                    out[i] = 0.0;
            }}
        }}
    """

    return code, "void gt_with_scalar(float *arr, float *out);"

@cache
def gte(x_numel: int, val: int | float) -> tuple[str, str]:
    code = f"""
        void gte_with_scalar(float *arr, float *out)
        {{
            for (int i=0; i<{x_numel}; i++) {{
                if (arr[i] >= {val})
                    out[i] = 1.0;
                else
                    out[i] = 0.0;
            }}
        }}
    """
    return code, "void gte_with_scalar(float *arr, float *out);"

@cache
def where(ndim: int) -> tuple[str, str]:
    """
    Generates a where kernel.
    out = (cond > 0) ? x : y
    """
    func_name = f"where_{ndim}d"
    cdef = f"void {func_name}(const float *cond, const float *x, const float *y, float *out, int *shape, int *c_stride, int *x_stride, int *y_stride, int *out_stride);"

    # Generate Nested Loops
    loops = ""
    indent = "    "
    for d in range(ndim):
        loops += f"{indent * (d + 1)}for (int i{d} = 0; i{d} < shape[{d}]; i{d}++) {{\n"

    # Calculate Indices
    body_indent = indent * (ndim + 1)

    # Calculate index for Condition
    c_idx_parts = [f"i{d}*c_stride[{d}]" for d in range(ndim)]
    calc_c_idx = " + ".join(c_idx_parts)

    # Calculate index for X (True branch)
    x_idx_parts = [f"i{d}*x_stride[{d}]" for d in range(ndim)]
    calc_x_idx = " + ".join(x_idx_parts)

    # Calculate index for Y (False branch)
    y_idx_parts = [f"i{d}*y_stride[{d}]" for d in range(ndim)]
    calc_y_idx = " + ".join(y_idx_parts)

    # Calculate index for Output
    out_idx_parts = [f"i{d}*out_stride[{d}]" for d in range(ndim)]
    calc_out_idx = " + ".join(out_idx_parts)

    # Close Loops
    closing_braces = ""
    for k in range(ndim - 1, -1, -1):
        depth = k + 1
        closing_braces += f"{indent * depth}}}\n"

    c_code = f"""
        void {func_name}(const float *cond, const float *x, const float *y, float *out, int *shape, int *c_stride, int *x_stride, int *y_stride, int *out_stride) {{
            {loops}
                {body_indent}int c_idx = {calc_c_idx};
                {body_indent}int x_idx = {calc_x_idx};
                {body_indent}int y_idx = {calc_y_idx};
                {body_indent}int out_idx = {calc_out_idx};

                {body_indent}out[out_idx] = (cond[c_idx] > 0) ? x[x_idx] : y[y_idx];
            {closing_braces}
        }}
    """
    return c_code, cdef

@cache
def cal_stride(first_inp_shape: tuple, cat_dim: int) -> int:
    stride = 1
    for i in range(cat_dim, len(first_inp_shape)):
        stride *= first_inp_shape[i]
    return stride

@cache
def cal_out_steps(cat_dim: int, first_inp_shape: tuple) -> int:
    steps = 1
    for i in range(cat_dim):
        steps *= first_inp_shape[i]
    return steps

@cache
def masked_fill(ndim: int) -> tuple[str, str]:
    """
    Generates a masked_fill kernel.
    """
    func_name = f"masked_fill_{ndim}d"
    cdef = f"void {func_name}(const float *x, const float *mask, float *out, int *shape, int *x_stride, int *mask_stride, int *out_stride, float fill_value);"

    # Generate Nested Loops (Iterate over Output Shape)
    loops = ""
    indent = "    "
    for d in range(ndim):
        loops += f"{indent * (d + 1)}for (int i{d} = 0; i{d} < shape[{d}]; i{d}++) {{\n"

    # Calculate Indices
    body_indent = indent * (ndim + 1)

    # Input X Index
    x_idx_parts = [f"i{d}*x_stride[{d}]" for d in range(ndim)]
    calc_x_idx = " + ".join(x_idx_parts)

    # Mask Index
    mask_idx_parts = [f"i{d}*mask_stride[{d}]" for d in range(ndim)]
    calc_mask_idx = " + ".join(mask_idx_parts)

    # Output Index
    out_idx_parts = [f"i{d}*out_stride[{d}]" for d in range(ndim)]
    calc_out_idx = " + ".join(out_idx_parts)

    # Close Loops
    closing_braces = ""
    for k in range(ndim - 1, -1, -1):
        depth = k + 1
        closing_braces += f"{indent * depth}}}\n"

    c_code = f"""
        void {func_name}(const float *x, const float *mask, float *out, int *shape, int *x_stride, int *mask_stride, int *out_stride, float fill_value) {{
            {loops}
                {body_indent}int x_idx = {calc_x_idx};
                {body_indent}int mask_idx = {calc_mask_idx};
                {body_indent}int out_idx = {calc_out_idx};

                // If mask is true (>0), fill with value. Else copy x.
                {body_indent}out[out_idx] = (mask[mask_idx] > 0) ? fill_value : x[x_idx];
            {closing_braces}
        }}
    """
    return c_code, cdef

@cache
def cmp_2d(mode: str) -> tuple[str, str]:
    match mode:
        case "<=":
            c_code = """
                #include <stdio.h>
                void cmp(float *x, float *y, float *out, int *out_shape, int *x_stride, int *y_stride, int *out_stride) {
                    for (int i=0; i<out_shape[0]; i++) {
                        for (int j=0; j<out_shape[1]; j++) {
                            int x_idx = i*x_stride[0] + j*x_stride[1];
                            int y_idx = i*y_stride[0] + j*y_stride[1];
                            out[i*out_stride[0] + j*out_stride[1]] = x[x_idx] <= y[y_idx] ? 1.0 : 0.0;
                        }
                    }
                }
            """
            return c_code, "void cmp(float *x, float *y, float *out, int *out_shape, int *x_stride, int *y_stride, int *out_stride);"

@cache
def uninitialized_memory() -> tuple[str, str]:
    c_code = """
        #include <stdlib.h>

        // dlgrad only supports float, hence it is ok to have the return type as float
        float *uninitialized_memory(size_t nbytes)
        {
            float *out = malloc(nbytes);
            if (out == NULL) {
                return NULL;
            }

            return out;
        }
    """

    return c_code, "float *uninitialized_memory(size_t nbytes);"

@cache
def initialized_memory() -> tuple[str, str]:
    c_code = """
        #include <stdlib.h>

        float *initialized_memory(size_t num, size_t size)
        {
            float *out = calloc(num, size);
            if (out == NULL) {
                return NULL;
            }

            return out;
        }
    """

    return c_code, "float *initialized_memory(size_t num, size_t size);"

@cache
def init_with_scalar() -> tuple[str, str]:
    c_code = """
        #include <stdlib.h>

        float *init_with_scalar(size_t nbytes, int numel, int scalar)
        {
            float *out = malloc(nbytes);
            if (out == NULL) {
                return NULL;
            }

            for (int i=0; i<numel; i++) {
                out[i] = scalar;
            }

            return out;
        }
    """

    return c_code, "float *init_with_scalar(size_t nbytes, int numel, int scalar);"

@cache
def free_ptr() -> tuple[str, str]:
    c_code = """
        #include <stdlib.h>

        void free_ptr(float *ptr)
        {
            free(ptr);
        }
    """

    return c_code, "void free_ptr(float *ptr);"

@cache
def arange(x_numel: int) -> tuple[str, str]:
    c_code = f"""
        #include <stdlib.h>

        void arange(float *out)
        {{
            for (int i=0; i<{x_numel}; i++) {{
                out[i] = i;
            }}
        }}
    """

    return c_code, "void arange(float *out);"

@cache
def full(x_numel: int, fill_value: int) -> tuple[str, str]:
    c_code = f"""
        #include <stdlib.h>

        void full(float *out)
        {{
            for (int i=0; i<{x_numel}; i++) {{
                out[i] = {fill_value};
            }}
        }}
    """

    return c_code, "void full(float *out);"


@cache
def uniform(x_numel: int, low: int, high: int) -> tuple[str, str]:
    c_code = f"""
        /*
        * pcg random number generation for c.
        *
        * copyright 2014 melissa o'neill <oneill@pcg-random.org>
        *
        * licensed under the apache license, version 2.0 (the "license");
        * you may not use this file except in compliance with the license.
        * you may obtain a copy of the license at
        *
        *     http://www.apache.org/licenses/license-2.0
        *
        * unless required by applicable law or agreed to in writing, software
        * distributed under the license is distributed on an "as is" basis,
        * without warranties or conditions of any kind, either express or implied.
        * see the license for the specific language governing permissions and
        * limitations under the license.
        *
        * for additional information about the pcg random number generation scheme,
        * including its license and other licensing options, visit
        *
        *     http://www.pcg-random.org
        */

        /*
        * this code is derived from the full c implementation, which is in turn
        * derived from the canonical c++ pcg implementation. the c++ version
        * has many additional features and is preferable if you can use c++ in
        * your project.
        */

        #include <inttypes.h>
        #include <stdlib.h>
        #include <time.h>
        #include <math.h>
        #include <fcntl.h>
        #include <unistd.h>
        #include <stdint.h>

        struct pcg_state_setseq_64 {{    // internals are *private*.
            uint64_t state;             // rng state.  all values are possible.
            uint64_t inc;               // controls which rng sequence (stream) is
                                        // selected. must *always* be odd.
        }};
        typedef struct pcg_state_setseq_64 pcg32_random_t;

        uint32_t pcg32_random_r(pcg32_random_t* rng)
        {{
            uint64_t oldstate = rng->state;
            rng->state = oldstate * 6364136223846793005ull + rng->inc;
            uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }}

        void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq)
        {{
            rng->state = 0u;
            rng->inc = (initseq << 1u) | 1u;
            pcg32_random_r(rng);
            rng->state += initstate;
            pcg32_random_r(rng);
        }}

        pcg32_random_t rng;

        int uniform(float *out)
        {{
            // int fd = open("/dev/random", o_rdonly);
            int fd = open("/dev/random", O_RDONLY);
            if (fd < 0) {{
                return -1;
            }}

            uint64_t seed;
            ssize_t bytes_read = read(fd, &seed, sizeof(seed));
            if (bytes_read != sizeof(seed)) {{
                return -1;
            }}

            close(fd);

            pcg32_srandom_r(&rng, seed, (intptr_t)&rng);

            for (int i= 0; i < {x_numel}; i++) {{
                double d = ldexp(pcg32_random_r(&rng), -32);
                float f = (float) d;

                if ({low}f == 0.0f && {high}f == 1.0f) {{
                    out[i] = f;
                }} else {{
                    out[i] = {low}f + ({high}f - {low}f) * f;
                }}
            }}

            return 0;
        }}
    """

    return c_code, "int uniform(float *out);"

@cache
def mnist_loader() -> tuple[str, str]:
    # todo: check if this works
    """
    uint8_t *temp_buf = (uint8_t*)malloc(out_size);
    if (!temp_buf) { free(out); fclose(fp); return null; }

    if (safe_fread(temp_buf, 1, out_size, fp) != out_size) {
        free(temp_buf); free(out); fclose(fp); return null;
    }

    // 3. cast to float in memory
    for (int i=0; i<out_size; i++) {
        out[i] = (float)temp_buf[i];
    }

    free(temp_buf);
    fclose(fp);
    """
    c_code = """
        #include <stdio.h>
        #include <stdlib.h>
        #include <arpa/inet.h>

        size_t safe_fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
            size_t ret = fread(ptr, size, nmemb, stream);
            if (ret != nmemb) {
                printf("Read error or unexpected EOF\\n");
            }
            return ret;
        }

        // https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/index.html
        float *mnist_images_loader(char *path, uint32_t magic_number)
        {
            FILE *fp = fopen(path, "rb");
            if (!fp) {
                printf("Failed to open file\\n");
                return NULL;
            }

            uint32_t magic_num, num_images, num_rows, num_cols;

            if (safe_fread(&magic_num, 1, 4, fp) != 4) {
                fclose(fp);
                return NULL;
            }

            magic_num = ntohl(magic_num);

            if (magic_num != magic_number) {
                printf("Invalid magic number\\n");
                fclose(fp);
                return NULL;
            }

            if (safe_fread(&num_images, 1, 4, fp) != 4) {
                fclose(fp);
                return NULL;
            }
            if (safe_fread(&num_rows, 1, 4, fp) != 4) {
                fclose(fp);
                return NULL;
            }
            if (safe_fread(&num_cols, 1, 4, fp) != 4) {
                fclose(fp);
                return NULL;
            }

            num_images = ntohl(num_images);
            num_rows = ntohl(num_rows);
            num_cols = ntohl(num_cols);

            int out_size = num_images * num_rows * num_cols;

            float *out = (float*)malloc(out_size * sizeof(float));
            if (!out) {
                printf("Failed to allocate memory\\n");
                fclose(fp);
                return NULL;
            }

            unsigned char pixel;
            for (int i=0; i<out_size; i++) {
                if (safe_fread(&pixel, 1, 1, fp) != 1) {
                    fclose(fp);
                    return NULL;
                }
                out[i] = (float)pixel;
            }

            fclose(fp);

            return out;
        }

        float *mnist_labels_loader(char *path, uint32_t magic_number)
        {
            FILE *fp = fopen(path, "rb");
            if (!fp) {
                printf("Failed to open file\\n");
                return NULL;
            }

            uint32_t magic_num, num_images;

            if (safe_fread(&magic_num, 1, 4, fp) != 4) {
                fclose(fp);
                return NULL;
            }

            magic_num = ntohl(magic_num);

            if (magic_num != magic_number) {
                printf("Invalid magic number\\n");
                fclose(fp);
                return NULL;
            }

            if (safe_fread(&num_images, 1, 4, fp) != 4) {
                fclose(fp);
                return NULL;
            }
            num_images = ntohl(num_images);

            int out_size = num_images;

            float *out = (float*)malloc(out_size * sizeof(float));
            if (!out) {
                printf("Failed to allocate memory\\n");
                fclose(fp);
                return NULL;
            }

            unsigned char labels;
            for (int i=0; i<out_size; i++) {
                if (safe_fread(&labels, 1, 1, fp) != 1) {
                    fclose(fp);
                    return NULL;
                }
                out[i] = (float)labels;
            }

            fclose(fp);

            return out;
        }
    """
    return c_code, "float *mnist_images_loader(char *path, uint32_t magic_number);float *mnist_labels_loader(char *path, uint32_t magic_number);"

@cache
def print_0d_tensor() -> tuple[str, str]:
    c_code = """
    #include <stdio.h>
    #include <stdbool.h>

    void print_tensor(float *x) {{
        printf("%f\\n", x[0]);
    }}
    """
    return c_code, "void print_tensor(float *x);"

@cache
def print_1d_tensor(shape: tuple, stride: tuple, numel: int) -> tuple[str, str]:
    w_trunc = "false"
    if (W := shape[0]) > 3 and W > 9:
        W = 3
        w_trunc = "true"
    c_code = f"""
        #include <stdio.h>
        #include <stdbool.h>

        void print_tensor(float *x) {{
            printf("[");
            for (int w=0; w<{W}; w++) {{
                printf("%f", x[w*{stride[0]}]);
                if (w != {W} - 1)
                    printf(", ");
                else {{
                    if ({w_trunc}) {{
                        printf(" ...");
                        for (int w={shape[0] - 3}; w<{shape[0]}; w++)
                            printf(" %f", x[w*{stride[0]}]);
                    }}
                }}
            }}
            printf("]\\n");
        }}
    """
    return c_code, "void print_tensor(float *x);"

@cache
def print_2d_tensor(shape: tuple, stride: tuple, numel: int) -> tuple[str, str]:
    h_trunc = "false"
    w_trunc = "false"
    if (H := shape[0]) > 3 and H > 9:
        H = 3
        h_trunc = "true"
    if (W := shape[1]) > 3 and W > 9:
        W = 3
        w_trunc = "true"
    c_code = f"""
        #include <stdio.h>
        #include <stdbool.h>

        void print_tensor(float *x) {{
            printf("[\\n");
            for (int h=0; h<{H}; h++) {{
                printf("  [");
                for (int w=0; w<{W}; w++) {{
                    printf("%f", x[h*{stride[0]} + w*{stride[1]}]);
                    if (w != {W} - 1)
                        printf(", ");
                    else {{
                        if ({w_trunc}) {{
                            printf(" ...");
                            for (int w={shape[1] - 3}; w<{shape[1]}; w++)
                                printf(" %f", x[h*{stride[0]} + w*{stride[1]}]);
                        }}
                    }}
                }}
                if (h != {H} - 1)
                    printf("],\\n");
                else {{
                    if ({h_trunc}) {{
                        printf("],\\n  ... \\n");
                    }} else {{
                        printf("]\\n");
                    }}
                }}
            }}
            printf("]\\n");
        }}
    """
    return c_code, "void print_tensor(float *x);"

@cache
def print_3d_tensor(shape: tuple, stride: tuple, numel: int) -> tuple[str, str]:
    c_trunc = "false"
    h_trunc = "false"
    w_trunc = "false"
    if (C := shape[0]) > 3 and C > 9:
        C = 3
        c_trunc = "true"
    if (H := shape[1]) > 3 and H > 9:
        H = 3
        h_trunc = "true"
    if (W := shape[2]) > 3 and W > 9:
        W = 3
        w_trunc = "true"
    c_code = f"""
        #include <stdio.h>
        #include <stdbool.h>

        void print_tensor(float *x) {{
            printf("[\\n");
            printf("    [");
            for (int c=0; c<{C}; c++) {{
                for (int h=0; h<{H}; h++) {{
                    printf("[");
                    for (int w=0; w<{W}; w++) {{
                        printf("%f", x[c*{stride[0]} + h*{stride[1]} + w*{stride[2]}]);
                        if (w != {W} - 1)
                            printf(", ");
                        else {{
                            if ({w_trunc}) {{
                                printf(" ...");
                                for (int w={shape[2] - 3}; w<{shape[2]}; w++)
                                    printf(" %f", x[c*{stride[0]} + h*{stride[1]} + w*{stride[2]}]);
                            }}
                        }}
                    }}
                    if (h != {H} - 1)
                        printf("], ");
                    else {{
                        if ({h_trunc}) {{
                            printf("], ... ");
                        }}
                    }}
                }}
                if (c != {C} - 1)
                    printf("]],\\n    [");
                else {{
                    printf("]],\\n");
                    if ({c_trunc})
                        printf("    ...\\n");
                }}
            }}
            printf("]\\n");
        }}
    """
    return c_code, "void print_tensor(float *x);"

@cache
def print_4d_tensor(shape: tuple, stride: tuple, numel: int) -> tuple[str, str]:
    h_trunc = "false"
    w_trunc = "false"
    c_trunc = "false"
    b_trunc = "false"
    if (B := shape[0]) > 3 and B > 9:
        B = 3
        b_trunc = "true"
    if (C := shape[1]) > 3 and C > 9:
        C = 3
        c_trunc = "true"
    if (H := shape[2]) > 3 and H > 9:
        H = 3
        h_trunc = "true"
    if (W := shape[3]) > 3 and W > 9:
        W = 3
        w_trunc = "true"
    c_code = f"""
        #include <stdio.h>
        #include <stdbool.h>

        void print_tensor(float *x) {{
            printf("[\\n");
            for (int b=0; b<{B}; b++) {{
                printf("  [\\n");
                printf("    [");
                for (int c=0; c<{C}; c++) {{
                    for (int h=0; h<{H}; h++) {{
                        printf("[");
                        for (int w=0; w<{W}; w++) {{
                            printf("%f", x[b*{stride[0]} + c*{stride[1]} + h*{stride[2]} + w*{stride[3]}]);
                            if (w != {W} - 1)
                                printf(", ");
                            else {{
                                if ({w_trunc}) {{
                                    printf(" ...");
                                    for (int w={shape[3] - 3}; w<{shape[3]}; w++)
                                        printf(" %f", x[b*{stride[0]} + c*{stride[1]} + h*{stride[2]} + w*{stride[3]}]);
                                }}
                            }}
                        }}
                        if (h != {H} - 1)
                            printf("], ");
                        else {{
                            if ({h_trunc}) {{
                                printf("], ... ");
                            }}
                        }}
                    }}
                    if (c != {C} - 1)
                        printf("]],\\n    [");
                    else {{
                        printf("]],\\n");
                        if ({c_trunc})
                            printf("    ...\\n");
                    }}
                }}
                if (b != {B} - 1)
                    printf("  ],\\n");
                else {{
                    if ({b_trunc})
                        printf("  ],\\n  ...\\n");
                    else
                        printf("  ]\\n");
                }}
            }}
            printf("]\\n");
        }}
    """
    return c_code, "void print_tensor(float *x);"

