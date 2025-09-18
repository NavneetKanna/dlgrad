from collections.abc import Generator
from functools import cache
from typing import Any


# NOTE: Assumes max 4D tensor
def n_gen() -> Generator[str, Any, None]:
    a = ["i", "j", "k", "l"]
    yield from a


@cache
def arithmetic(x_shape: tuple, x_stride: tuple, y_shape: tuple, y_stride: tuple, op: str) -> tuple[str, str]:  # noqa: C901
    gen = n_gen()

    code  = f"""
    #include <stdlib.h>

    void {op}(float *x, float *y, float *out) {{
        int x_off, y_off;
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

    if not y_shape or all([1==i for i in y_shape]): # scalars
        ts = "y_off = 0;"
    else:
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
        case "divv":
            code += """
                out[x_off] = x[x_off] / y[y_off];
            """

    for i in range(len(var_str)):
        code += "}\n"

    code += "}\n"

    return code, f"void {op}(float *x, float *y, float *out);"

@cache
def max_backward(x_shape: tuple, x_stride: tuple, x_numel: int, dim: int) -> tuple[str, str]:
    code  = f"""
        void max_backward(float *x, float *out, float *max_with_1s) {{
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
                    for (int i = 0; i < stride_dim; i++) {{
                        float val = x[j + i];
                        float val2 = out[out_start + i];
                        if (val == val2) {{
                            max_with_1s[j+i] = 1.0f;
                        }}
                    }}
                }} else {{
                    for (int i = 0; i < stride_dim; i++) {{
                        float val = x[j + i];
                        float val2 = out[out_start + i];
                        if (val == val2) {{
                            max_with_1s[j+i] = 1.0f;
                        }}
                    }}
                }}
            }}
        }}
    """

    return code, "void max_backward(float *x, float *out, float *max_with_1s);"

@cache
def max(x_shape: tuple, x_stride: tuple, x_numel: int, dim: int) -> tuple[str, str]:
    if dim == -1:
        code = f"""
            void max(float *x, float *out) {{
                float m = 0.0;
                for (int i=0; i<{x_numel}; i++) {{
                    if (x[i] > m) {{
                        m = x[i];
                    }}
                }}
                out[0] = m;
            }}
        """

        return code, "void max(float *x, float *out);"

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

@cache
def sum(x_shape: tuple, x_stride: tuple, x_numel: int, dim: int) -> tuple[str, str]:
    if dim == -1:
        code = f"""
            void sum(float *x, float *out) {{
                float s = 0.0;
                for (int i=0; i<{x_numel}; i++) {{
                    s += x[i];
                }}
                out[0] = s;
            }}
        """

        return code, "void sum(float *x, float *out);"

    code  = f"""
    void sum(float *x, float *out) {{
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
                // sum
                for (int i = 0; i < stride_dim; i++) {{
                    float val = x[j + i];
                    out[out_start + i] += val;
                }}
            }}
        }}
    }}
    """

    return code, "void sum(float *x, float *out);"

@cache
def transpose_given_axes(x_shape: tuple, axes: tuple) -> list:
    ax = list(range(len(x_shape)))
    # replace elements at positions in axes with elements of axes reversed
    for i, pos in enumerate(axes):
        ax[pos] = axes[-(i+1)]
    return ax

@cache
def transpose(x_shape: tuple, x_stride: tuple,  out_stride: tuple, x_numel: int, axes: tuple) -> tuple[str, str]:
    gen = n_gen()

    ax = transpose_given_axes(x_shape, axes)

    code = """
    void transpose(float *x, float *out) {
    int out_idx = 0;
    int x_idx = 0;
    """

    var_str = []
    for i in x_shape:
        var = next(gen)
        var_str.append(var)
        code += f"""
        for (int {var}=0; {var}<{i}; {var}++) {{
        """

    ts = "out_idx = "
    for i, v in zip(out_stride, ax):
        ts += f"{i}*{var_str[v]} + "
    ts = ts[:-3]
    ts += ";"

    code += ts

    code += "\n"

    ts = "x_idx = "
    for i, v in zip(x_stride, var_str):
        ts += f"{i}*{v} + "
    ts = ts[:-3]
    ts += ";"

    code += ts

    code += "\n"

    code += """
    out[out_idx] = x[x_idx];
    """

    for i in range(len(var_str)):
        code += "}\n"

    code += "}\n"

    return code, "void transpose(float *x, float *out);"

@cache
def utils(x_numel: int, func: str) -> tuple[str, str]:
    match func:
        case "neg":
            code = f"""
            #include <math.h>

            void neg(float *x, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = -1 * x[i];
                }}
            }}
            """

            return code, "void neg(float *x, float *out);"
        case "exp":
            code = f"""
            #include <math.h>

            void cexp(float *x, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = exp(x[i]);
                }}
            }}
            """

            return code, "void cexp(float *x, float *out);"
        case "log":
            code = f"""
            #include <math.h>

            void clog(float *x, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = log(x[i]);
                }}
            }}
            """

            return code, "void clog(float *x, float *out);"
        case "pow":
            code = f"""
            #include <math.h>

            void c_pow(float *x, float *out, int val)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = powf(x[i], val);
                }}
            }}
            """

            return code, "void c_pow(float *x, float *out, int val);"
        case "sqrt":
            code = f"""
            #include <math.h>

            void csqrt(float *x, float *out)
            {{
                for (int i=0; i<{x_numel}; i++) {{
                    out[i] = sqrtf(x[i]);
                }}
            }}
            """

            return code, "void csqrt(float *x, float *out);"

@cache
def matmul(x_shape: tuple, y_shape: tuple, x_stride: tuple, y_stride: tuple) -> tuple[str, str]:
    c_code = f"""
        void matmul(float *x, float *y, float *out) {{
            float sum = 0.0;
            for (int i=0; i<{x_shape[0]}; i++) {{
                for (int j=0; j<{y_shape[1]}; j++) {{
                    sum = 0.0;
                    for (int k=0; k<{y_shape[0]}; k++) {{
                        sum += x[i*{x_stride[0]} + k*{x_stride[1]}] * y[k*{y_stride[0]} + j*{y_stride[1]}];
                    }}
                    out[i*{y_shape[1]}+ j] = sum;
                }}
            }}
        }}
        """

    return c_code, "void matmul(float *x, float *y, float *out);"

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
def argmax(x_shape: tuple, axis: int) -> tuple[str, str]:
    code = f"""
        void argmax2d(float *x, float *out)
        {{
            int rows = {x_shape[0]};
            int cols = {x_shape[1]};
            if ({axis} == 1) {{
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
            }} else if ({axis} == 0) {{
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
            }} else {{
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
def eqt(x_numel: int) -> tuple[str, str]:
    code = f"""
        void eqt(float *x, float *y, float *out)
        {{
            for (int i=0; i<{x_numel}; i++) {{
                if (x[i] == y[i]) {{
                    out[i] = 1.0;
                }} else {{
                    out[i] = 0.0;
                }}
            }}
        }}
    """

    return code, "void eqt(float *x, float *y, float *out);"

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
    c_code = """
        #include <stdio.h>
        #include <stdlib.h>
        #include <arpa/inet.h>

        // https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/index.html
        float *mnist_images_loader(char *path, uint32_t magic_number)
        {
            FILE *fp = fopen(path, "rb");
            if (!fp) {
                printf("Failed to open file\\n");
                return NULL;
            }

            uint32_t magic_num, num_images, num_rows, num_cols;

            fread(&magic_num, 1, 4, fp);

            magic_num = ntohl(magic_num);

            if (magic_num != magic_number) {
                printf("Invalid magic number\\n");
                fclose(fp);
                return NULL;
            }

            fread(&num_images, 1, 4, fp);
            fread(&num_rows, 1, 4, fp);
            fread(&num_cols, 1, 4, fp);
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
                fread(&pixel, 1, 1, fp);
                out[i] = (float)pixel / 255.0f;
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

            fread(&magic_num, 1, 4, fp);

            magic_num = ntohl(magic_num);

            if (magic_num != magic_number) {
                printf("Invalid magic number\\n");
                fclose(fp);
                return NULL;
            }

            fread(&num_images, 1, 4, fp);
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
                fread(&labels, 1, 1, fp);
                out[i] = (float)labels;
            }

            fclose(fp);

            return out;
        }
    """
    return c_code, "float *mnist_images_loader(char *path, uint32_t magic_number);float *mnist_labels_loader(char *path, uint32_t magic_number);"


"""
dim = 1

(4, 3, 2)
(6, 2, 1)

[[[0.40329522 0.7129083 ]
  [0.2902877  0.9216182 ]
  [0.73256075 0.8372984 ]]

 [[0.59929603 0.36998346]
  [0.55582845 0.17390463]
  [0.05803654 0.39682135]]

 [[0.7792665  0.47508177]
  [0.20465787 0.48576224]
  [0.01099625 0.86618644]]

 [[0.15955642 0.7351426 ]
  [0.49416125 0.13069092]
  [0.07697563 0.8316998 ]]]

#include <stdlib.h>

void maxx(float *x, float *out) {
    int shape_dim = 3;
    int stride_dim = 2;
    int numel = 24;

    int out_start = 0;
    for (int j = 0; j < numel; j += stride_dim) {
        if ((j % (stride_dim * shape_dim)) == 0) {
            if (j != 0) {
                out_start += stride_dim;
            } else {
                out_start = 0;
            }
            // copy
            for (int i = 0; i < stride_dim; i++) {
                out[out_start + i] = x[j + i];
            }
        } else {
            // max
            for (int i = 0; i < stride_dim; i++) {
                float val = x[j + i];
                if (val > out[out_start + i]) {
                    out[out_start + i] = val;
                }
            }
        }
    }
}

at loop index 0,
out_start=0, j=0
copies [0.40329522 0.7129083] into out[0] and out[1]

at loop index 2,
out_start=0, j=2
find max b/w x[2] and out[0] and b/w x[3] and out[1], if x[2] or x[3] is > out[0] or out[1], then replace

at loop index 4,
out_start=0, j=4
find max b/w x[4] and out[0] and b/w x[5] and out[1], if x[4] or x[5] is > out[0] or out[1], then replace

at loop index 6,
out_start=2, j=6
copies [0.59929603 0.36998346] into out[2] and out[3]

at loop index 8,
out_start=2, j=8
find max b/w x[8] and out[2] and b/w x[9] and out[3], if x[8] or x[9] is > out[2] or out[3], then replace

and so on.

This applies to any shape and any dim. This algo also ensures that we are avoiding cache miss when trying to find
max or sum along a dim, by looping through the input and output array's sequentially.
"""
