from collections.abc import Generator
from functools import cache
from typing import Any

from dlgrad.dtype import Scalar


# NOTE: Assumes max 4D tensor
def n_gen() -> Generator[str, Any, None]:
    a = ["i", "j", "k", "l"]
    yield from a

cache
def arithmetic(x_shape: tuple, x_stride: tuple, y_shape: tuple, y_stride: tuple, op: str) -> tuple[str, str]:  # noqa: C901
    gen = n_gen()

    c_code = f"""
        #include <stdlib.h>
        void {op}(float *x, float *y, float *out) {{
    """
    var_str = []
    for i in x_shape:
        var = next(gen)
        var_str.append(var)
        c_code += f"""
        for (int {var}=0; {var}<{i}; {var}++) {{
        """

    if not y_shape or all([1==i for i in y_shape]): # scalar
        yo = "0"
    else:
        yo = ""
        for j, k, l in zip(y_shape, var_str, y_stride):  # noqa: E741
            if j == 1:
                yo += f"0*{k} + "
            else:
                yo += f"{l}*{k} + "
        yo = yo[:-3]

    ts = ""
    for i, v in zip(x_stride, var_str):
        ts += f"{i}*{v} + "
    ts = ts[:-3]

    if not x_shape:
        ts = "0"

    match op:
        case "add":
            c_code += f"""
                out[{ts}] = x[{ts}] + y[{yo}];
            """
        case "sub":
            c_code += f"""
                out[{ts}] = x[{ts}] - y[{yo}];
            """
        case "mul":
            c_code += f"""
                out[{ts}] = x[{ts}] * y[{yo}];
            """
        case "divv":
            c_code += f"""
                out[{ts}] = x[{ts}] / y[{yo}];
            """

    for i in range(len(var_str)):
        c_code += "}\n"

    c_code += "}\n"

    return c_code, f"void {op}(float *x, float *y, float *out);"

@cache
def reduce(x_numel: int, op: str) -> tuple[str, str]:
    if op == "sum":
        op1 = "float val = 0;"
        op2 = "val += v;"
    else:
        op1 = "float val = -FLT_MAX;"
        op2 = "if (v > val)\nval = v;"
    c_code = f"""
        #include <stdlib.h>
        #include <float.h>
        void {op}(const float *x, float *out) {{
            {op1}
            for (size_t i=0; i<{x_numel}; i++) {{
                float v = x[i];
                {op2}
            }}
            out[0] = val;
        }}
    """
    return c_code, f"void {op}(const float *x, float *out);"

@cache
def reduce_4d(x_shape: tuple, x_stride: tuple, out_stride: tuple, x_numel: int, dim: int, op: str) -> tuple[str, str]:
    if op == "sum":
        op1 = "float val = 0;"
        op2 = "val += v;"
    else:
        op1 = "float val = -FLT_MAX;"
        op2 = "if (v > val)\nval = v;"
    if dim == 0:
        c_code = f"""
            #include <stdlib.h>
            #include <float.h>
            void {op}(const float *x, float *out) {{
                for (size_t C = 0; C < {x_shape[1]}; C++) {{
                    for (size_t H = 0; H < {x_shape[2]}; H++) {{
                        for (size_t W = 0; W < {x_shape[3]}; W++) {{
                            {op1}
                            for (size_t B=0; B<{x_shape[0]}; B++) {{
                                float v = x[B*{x_stride[0]} + C*{x_stride[1]} + H*{x_stride[2]} + W];
                                {op2}
                            }}
                            out[C*{out_stride[0]} + H*{out_stride[1]} + W] = val;
                        }}
                    }}
                }}
            }}
        """
        return c_code, f"void {op}(const float *x, float *out);"
    elif dim == 1:
        c_code = f"""
            #include <stdlib.h>
            #include <float.h>
            void {op}(const float *x, float *out) {{
                for (size_t B=0; B<{x_shape[0]}; B++) {{
                    for (size_t H=0; H<{x_shape[2]}; H++) {{
                        for (size_t W=0; W<{x_shape[3]}; W++) {{
                            {op1}
                            for (size_t C=0; C<{x_shape[1]}; C++) {{
                                float v = x[B*{x_stride[0]} + C*{x_stride[1]} + H*{x_stride[2]} + W];
                                {op2}
                            }}
                            out[B*{out_stride[0]} + H*{out_stride[1]} + W] = val;
                        }}
                    }}
                }}
            }}
        """
        return c_code, f"void {op}(const float *x, float *out);"
    elif dim == 2:
        c_code = f"""
            #include <stdlib.h>
            #include <float.h>
            void {op}(const float *x, float *out) {{
                for (size_t B=0; B<{x_shape[0]}; B++) {{
                    for (size_t C=0; C<{x_shape[1]}; C++) {{
                        for (size_t W=0; W<{x_shape[3]}; W++) {{
                            {op1}
                            for (size_t H=0; H<{x_shape[2]}; H++) {{
                                float v = x[B*{x_stride[0]} + C*{x_stride[1]} + H*{x_stride[2]} + W];
                                {op2}
                            }}
                            out[B*{out_stride[0]} + C*{out_stride[1]} + W] = val;
                        }}
                    }}
                }}
            }}
        """
        return c_code, f"void {op}(const float *x, float *out);"
    elif dim == 3:
        c_code = f"""
            #include <stdlib.h>
            #include <float.h>
            void {op}(const float *x, float *out) {{
                for (size_t B=0; B<{x_shape[0]}; B++) {{
                    for (size_t C=0; C<{x_shape[1]}; C++) {{
                        for (size_t H=0; H<{x_shape[2]}; H++) {{
                            const float *ptr = x + B*{x_stride[0]} + C*{x_stride[1]} + H*{x_stride[2]};
                            {op1}
                            for (size_t W=0; W<{x_shape[3]}; W++) {{
                                float v = ptr[W];
                                {op2}
                            }}
                            out[B*{out_stride[0]} + C*{out_stride[1]} + H] = val;
                        }}
                    }}
                }}
            }}
        """
        return c_code, f"void {op}(const float *x, float *out);"
    elif dim == -1:
        return reduce(x_numel, op)

@cache
def reduce_3d(x_shape: tuple, x_stride: tuple, out_stride: tuple, x_numel: int, dim: int, op: str) -> tuple[str, str]:
    if op == "sum":
        op1 = "float val = 0;"
        op2 = "val += v;"
    else:
        op1 = "float val = -FLT_MAX;"
        op2 = "if (v > val)\nval = v;"
    if dim == 0:
        c_code = f"""
            #include <stdio.h>
            #include <float.h>
            void {op}(const float *x, float *out) {{
                    for (size_t H=0; H<{x_shape[1]}; H++) {{
                        for (size_t W=0; W<{x_shape[2]}; W++) {{
                            {op1}
                            for (size_t C=0; C<{x_shape[0]}; C++) {{
                                float v = x[C*{x_stride[0]} + H*{x_stride[1]} + W];
                                {op2}
                            }}
                            out[H*{out_stride[0]} + W] = val;
                        }}
                    }}
            }}
        """
        return c_code, f"void {op}(const float *x, float *out);"
    elif dim == 1:
        c_code = f"""
            #include <stdio.h>
            #include <float.h>
            void {op}(const float *x, float *out) {{
                    for (size_t C=0; C<{x_shape[0]}; C++) {{
                        for (size_t W=0; W<{x_shape[2]}; W++) {{
                            {op1}
                            for (size_t H=0; H<{x_shape[1]}; H++) {{
                                float v = x[C*{x_stride[0]} + H*{x_stride[1]} + W];
                                {op2}
                            }}
                            out[C*{out_stride[0]} + W] = val;
                        }}
                    }}
            }}
        """
        return c_code, f"void {op}(const float *x, float *out);"
    elif dim == 2:
        c_code = f"""
            #include <stdio.h>
            #include <float.h>
            void {op}(const float *x, float *out) {{
                for (size_t C=0; C<{x_shape[0]}; C++) {{
                    for (size_t H=0; H<{x_shape[1]}; H++) {{
                        const float *ptr = x + C*{x_stride[0]} + H*{x_stride[1]};
                        {op1}
                        for (size_t W=0; W<{x_shape[2]}; W++) {{
                            float v = ptr[W];
                            {op2}
                        }}
                        out[C*{out_stride[0]} + H] = val;
                    }}
                }}
            }}
        """
        return c_code, f"void {op}(const float *x, float *out);"
    elif dim == -1:
        return reduce(x_numel, op)

@cache
def reduce_2d(x_shape: tuple, x_stride: tuple, out_stride: tuple, x_numel: int, dim: int, op: str) -> tuple[str, str]:
    if op == "sum":
        op1 = "float val = 0;"
        op2 = "val += v;"
    else:
        op1 = "float val = -FLT_MAX;"
        op2 = "if (v > val)\nval = v;"
    if dim == 0:
        c_code = f"""
            #include <stdio.h>
            #include <float.h>
            void {op}(const float *x, float *out) {{
                for (size_t W=0; W<{x_shape[1]}; W++) {{
                    {op1}
                    for (size_t H=0; H<{x_shape[0]}; H++) {{
                        float v = x[H*{x_stride[0]} + W];
                        {op2}
                    }}
                    out[W] = val;
                }}
            }}
        """
        return c_code, f"void {op}(const float *x, float *out);"
    elif dim == 1:
        c_code = f"""
            #include <stdio.h>
            #include <float.h>
            void {op}(const float *x, float *out) {{
                for (size_t H=0; H<{x_shape[0]}; H++) {{
                    const float *ptr = x + H*{x_stride[0]};
                    {op1}
                    for (size_t W=0; W<{x_shape[1]}; W++) {{
                        float v = ptr[W];
                        {op2}
                    }}
                    out[H] = val;
                }}
            }}
        """
        return c_code, f"void {op}(const float *x, float *out);"
    elif dim == -1:
        return reduce(x_numel, op)

@cache
def mean(x_shape: tuple, x_stride: tuple, x_numel: int, dim: int, out_numel: int) -> tuple[str, str]:
    if dim == -1:
        code = f"""
            void mean(float *x, float *out) {{
                float s = 0.0;
                for (int i=0; i<{x_numel}; i++) {{
                    s += x[i];
                }}
                out[0] = s / {x_numel};
            }}
        """

        return code, "void mean(float *x, float *out);"

    code  = f"""
    void mean(float *x, float *out) {{
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
                // mean
                for (int i = 0; i < stride_dim; i++) {{
                    float val = x[j + i];
                    out[out_start + i] += val;
                }}
            }}
        }}

        for (int i=0; i<{out_numel}; i++) {{
            out[i] = out[i] / shape_dim;
        }}
    }}
    """

    return code, "void mean(float *x, float *out);"

@cache
def transpose_4d_23(out_shape: tuple, out_stride: tuple, x_stride: tuple) -> tuple[str, str]:
    c_code = f"""
        void transpose(float *x, float *out) {{
            for (int B=0; B<{out_shape[0]}; B++) {{
                for (int C=0; C<{out_shape[1]}; C++) {{
                    for (int H=0; H<{out_shape[2]}; H++) {{
                        for (int W=0; W<{out_shape[3]}; W++) {{
                            int out_idx = B*{out_stride[0]} + C*{out_stride[1]} + H*{out_stride[2]} + W*{out_stride[3]};
                            int x_idx = B*{x_stride[0]} + C*{x_stride[1]} + W*{x_stride[2]} + H*{x_stride[3]};
                            out[out_idx] = x[x_idx];
                        }}
                    }}
                }}
            }}
        }}
    """
    return c_code, "void transpose(float *x, float *out);"

@cache
def transpose_4d_12(out_shape: tuple, out_stride: tuple, x_stride: tuple) -> tuple[str, str]:
    c_code = f"""
        #include <string.h>

        void transpose(float *x, float *out) {{
            for (int B=0; B<{out_shape[0]}; B++) {{
                for (int C=0; C<{out_shape[1]}; C++) {{
                    for (int H=0; H<{out_shape[2]}; H++) {{
                        int out_idx = B*{out_stride[0]} + C*{out_stride[1]} + H*{out_stride[2]};
                        int x_idx = B*{x_stride[0]} + H*{x_stride[1]} + C*{x_stride[2]};
                        memcpy(&out[out_idx], &x[x_idx], {out_shape[3]}*sizeof(float));
                    }}
                }}
            }}
        }}
    """
    return c_code, "void transpose(float *x, float *out);"

@cache
def transpose_3d_01(x_shape: tuple, out_shape: tuple, x_stride: tuple,  out_stride: tuple, x_numel: int) -> tuple[str, str]:
    c_code = f"""
        void transpose(float *x, float *out) {{
            int out_idx = 0;
            int x_idx = 0;
            for (int i=0; i<{out_shape[0]}; i++) {{
                for (int j=0; j<{out_shape[1]}; j++) {{
                    for (int k=0; k<{out_shape[2]}; k++) {{
                        x_idx = i*{x_stride[1]} + j*{x_stride[0]} + k*{x_stride[2]};
                        out_idx = i*{out_stride[0]} + j*{out_stride[1]} + k*{out_stride[2]};
                        out[out_idx] = x[x_idx];
                    }}
                }}
            }}
        }}
    """
    return c_code, "void transpose(float *x, float *out);"

@cache
def transpose_3d_12(x_shape: tuple, out_shape: tuple, x_stride: tuple,  out_stride: tuple, x_numel: int) -> tuple[str, str]:
    c_code = f"""
        void transpose(float *x, float *out) {{
            int out_idx = 0;
            int x_idx = 0;
            for (int k=0; k<{x_shape[0]}; k++) {{
                int out_idx = 0;
                int x_idx = 0;
                for (int i=0; i<{x_shape[1]}; i++) {{
                    for (int j=0; j<{x_shape[2]}; j++) {{
                        out_idx = k*{out_stride[0]} + {out_stride[1]}*j + {out_stride[2]}*i;
                        x_idx = k*{x_stride[0]} + {x_stride[1]}*i + {x_stride[2]}*j;
                        out[out_idx] = x[x_idx];
                    }}
                }}
            }}
        }}
    """
    return c_code, "void transpose(float *x, float *out);"

@cache
def transpose_2d(x_shape: tuple, x_stride: tuple,  out_stride: tuple, x_numel: int) -> tuple[str, str]:
    c_code = f"""
        void transpose(float *x, float *out) {{
            int out_idx = 0;
            int x_idx = 0;
            for (int i=0; i<{x_shape[0]}; i++) {{
                for (int j=0; j<{x_shape[1]}; j++) {{
                    out_idx = {out_stride[0]}*j + {out_stride[1]}*i;
                    x_idx = {x_stride[0]}*i + {x_stride[1]}*j;
                    out[out_idx] = x[x_idx];
                }}
            }}
        }}
    """
    return c_code, "void transpose(float *x, float *out);"

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
def where(x_numel: int, inp: bool, other: bool) -> tuple[str, str]:
    # if inp or other is True it is scalar
    s = ""

    code = f"""
        void where(float *arr, float *out, float *inp, float *other)
        {{
            for (int i=0; i<{x_numel}; i++) {{
                if (arr[i] > 0.0)
    """
    code = s + code
    if inp:
        ss = "out[i] = inp[0];"
    else:
        ss = "out[i] = inp[i];"

    code += ss

    code += "\nelse\n"

    if other:
        ss = "out[i] = other[0];"
    else:
        ss = "out[i] = other[i];"

    code += ss

    code += "}"
    code += "}"

    return code, "void where(float *arr, float *out, float *inp, float *other);"

@cache
def eqt(x_numel: int, y_scalar: bool, x_shape: tuple, x_stride: tuple, y_shape: tuple, y_stride: tuple, x_dim: int) -> tuple[str, str]:  # noqa: C901
    if y_scalar:
        code = f"""
            void eqt(float *x, float *y, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    if (x[i] == y[0]) {{
                        out[i] = 1.0;
                    }} else {{
                        out[i] = 0.0;
                    }}
                }}
            }}
        """
        return code, "void eqt(float *x, float *y, float *out);"

    if x_dim == 4:
        c = ""
        for d, i, j in zip(["B", "C", "H", "W"], y_shape, y_stride):
            if i != 1:
                c += f"{d}*{j} + "

        c = c[:-3] if c else "0"

        c_code = f"""
            #include <stdlib.h>
            void eqt(float *x, float *y, float *out)
            {{
                for (size_t B = 0; B < {x_shape[0]}; B++) {{
                    for (size_t C = 0; C < {x_shape[1]}; C++) {{
                        for (size_t H = 0; H < {x_shape[2]}; H++) {{
                            for (size_t W=0; W<{x_shape[3]}; W++) {{
                                int x_idx = B*{x_stride[0]} + C*{x_stride[1]} + H*{x_stride[2]} + W*{x_stride[3]};
                                int y_idx = {c};

                                if (x[x_idx] == y[y_idx]) {{
                                    out[x_idx] = 1.0;
                                }} else {{
                                    out[x_idx] = 0.0;
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        """
        return c_code, "void eqt(float *x, float *y, float *out);"
    elif x_dim == 3:
        c = ""
        for d, i, j in zip(["C", "H", "W"], y_shape, y_stride):
            if i != 1:
                c += f"{d}*{j} + "

        c = c[:-3] if c else "0"

        c_code = f"""
            #include <stdlib.h>
            void eqt(float *x, float *y, float *out)
            {{
                for (size_t C = 0; C < {x_shape[0]}; C++) {{
                    for (size_t H = 0; H < {x_shape[1]}; H++) {{
                        for (size_t W=0; W<{x_shape[2]}; W++) {{
                            int x_idx = C*{x_stride[0]} + H*{x_stride[1]} + W*{x_stride[2]};
                            int y_idx = {c};

                            if (x[x_idx] == y[y_idx]) {{
                                out[x_idx] = 1.0;
                            }} else {{
                                out[x_idx] = 0.0;
                            }}
                        }}
                    }}
                }}
            }}
        """
        return c_code, "void eqt(float *x, float *y, float *out);"
    elif x_dim == 2:
        c = ""
        for d, i, j in zip(["H", "W"], y_shape, y_stride):
            if i != 1:
                c += f"{d}*{j} + "

        c = c[:-3] if c else "0"

        c_code = f"""
            #include <stdlib.h>
            void eqt(float *x, float *y, float *out)
            {{
                for (size_t H = 0; H < {x_shape[0]}; H++) {{
                    for (size_t W=0; W<{x_shape[1]}; W++) {{
                        int x_idx = H*{x_stride[0]} + W*{x_stride[1]};
                        int y_idx = {c};

                        if (x[x_idx] == y[y_idx]) {{
                            out[x_idx] = 1.0;
                        }} else {{
                            out[x_idx] = 0.0;
                        }}
                    }}
                }}
            }}
        """
        return c_code, "void eqt(float *x, float *y, float *out);"

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
def masked_fill_4d(out_shape: tuple, out_stride: tuple, x_stride: tuple, mask_stride: tuple, val: Scalar) -> tuple[str, str]:
    if val == float('inf'):
        val = 'FLT_MAX'
    elif val == float('-inf'):
        val = '-FLT_MAX'

    c_code = f"""
        #include <math.h>
        #include <float.h>

        void masked_fill(float *x, float *y, float *out) {{
            for (int i=0; i<{out_shape[0]}; i++) {{
                for (int j=0; j<{out_shape[1]}; j++) {{
                    for (int k=0; k<{out_shape[2]}; k++) {{
                        for (int l=0; l<{out_shape[3]}; l++) {{
                            int x_idx = i*{x_stride[0]} + j*{x_stride[1]} + k*{x_stride[2]} + l*{x_stride[3]};
                            int y_idx = i*{mask_stride[0]} + j*{mask_stride[1]} + k*{mask_stride[2]} + l*{mask_stride[3]};
                            int out_idx = i*{out_stride[0]} + j*{out_stride[1]} + k*{out_stride[2]} + l*{out_stride[3]};

                            out[out_idx] = y[y_idx] > 0 ? {val}: x[x_idx];
                        }}
                    }}
                }}
            }}
        }}
    """
    return c_code, "void masked_fill(float *x, float *y, float *out);"


def masked_fill_3d(out_shape: tuple, out_stride: tuple, x_stride: tuple, y_stride: tuple, val: Scalar) -> tuple[str, str]:
    if val == float('inf'):
        val = 'FLT_MAX'
    elif val == float('-inf'):
        val = '-FLT_MAX'

    c_code = f"""
        #include <math.h>
        #include <float.h>

        void masked_fill(float *x, float *y, float *out) {{
            for (int i=0; i<{out_shape[0]}; i++) {{
                for (int j=0; j<{out_shape[1]}; j++) {{
                    for (int k=0; k<{out_shape[2]}; k++) {{
                        int x_idx = i*{x_stride[0]} + j*{x_stride[1]} + k*{x_stride[2]};
                        int y_idx = i*{y_stride[0]} + j*{y_stride[1]} + k*{y_stride[2]};
                        out[i*{out_stride[0]} + j*{out_stride[1]} + k*{out_stride[2]}] = y[y_idx] > 0 ? {val}: x[x_idx] ;
                    }}
                }}
            }}
        }}
    """
    return c_code, "void masked_fill(float *x, float *y, float *out);"

@cache
def cmp_2d(mode: str, out_shape: tuple, out_stride: tuple, x_shape: tuple, x_stride: tuple, y_stride: tuple) -> tuple[str, str]:
    match mode:
        case "<=":
            c_code = f"""
                #include <stdio.h>
                void cmp(float *x, float *y, float *out) {{
                    for (int i=0; i<{out_shape[0]}; i++) {{
                        for (int j=0; j<{out_shape[1]}; j++) {{
                            int x_idx = i*{x_stride[0]} + j*{x_stride[1]};
                            int y_idx = i*{y_stride[0]} + j*{y_stride[1]};
                            out[i*{out_stride[0]} + j*{out_stride[1]}] = x[x_idx] <= y[y_idx] ? 1.0 : 0.0;
                        }}
                    }}
                }}
            """
            return c_code, "void cmp(float *x, float *y, float *out);"

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
        * PCG Random Number Generation for C.
        *
        * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
        *
        * Licensed under the Apache License, Version 2.0 (the "License");
        * you may not use this file except in compliance with the License.
        * You may obtain a copy of the License at
        *
        *     http://www.apache.org/licenses/LICENSE-2.0
        *
        * Unless required by applicable law or agreed to in writing, software
        * distributed under the License is distributed on an "AS IS" BASIS,
        * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        * See the License for the specific language governing permissions and
        * limitations under the License.
        *
        * For additional information about the PCG random number generation scheme,
        * including its license and other licensing options, visit
        *
        *     http://www.pcg-random.org
        */

        /*
        * This code is derived from the full C implementation, which is in turn
        * derived from the canonical C++ PCG implementation. The C++ version
        * has many additional features and is preferable if you can use C++ in
        * your project.
        */

        #include <inttypes.h>
        #include <stdlib.h>
        #include <time.h>
        #include <math.h>
        #include <fcntl.h>
        #include <unistd.h>
        #include <stdint.h>

        struct pcg_state_setseq_64 {{    // Internals are *Private*.
            uint64_t state;             // RNG state.  All values are possible.
            uint64_t inc;               // Controls which RNG sequence (stream) is
                                        // selected. Must *always* be odd.
        }};
        typedef struct pcg_state_setseq_64 pcg32_random_t;

        uint32_t pcg32_random_r(pcg32_random_t* rng)
        {{
            uint64_t oldstate = rng->state;
            rng->state = oldstate * 6364136223846793005ULL + rng->inc;
            uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }}

        void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq)
        {{
            rng->state = 0U;
            rng->inc = (initseq << 1u) | 1u;
            pcg32_random_r(rng);
            rng->state += initstate;
            pcg32_random_r(rng);
        }}

        pcg32_random_t rng;

        int uniform(float *out)
        {{
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

                if ({low} == 0.0f && {high} == 1.0f) {{
                    out[i] = f;
                }} else {{
                    out[i] = {low} + ({high} - {low}) * f;
                }}
            }}

            return 0;
        }}
    """

    return c_code, "int uniform(float *out);"

@cache
def mnist_loader() -> tuple[str, str]:
    # TODO: Check if this works
    """
    uint8_t *temp_buf = (uint8_t*)malloc(out_size);
    if (!temp_buf) { free(out); fclose(fp); return NULL; }

    if (safe_fread(temp_buf, 1, out_size, fp) != out_size) {
        free(temp_buf); free(out); fclose(fp); return NULL;
    }

    // 3. Cast to float in memory
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

