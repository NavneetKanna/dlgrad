import gzip
import itertools
import os
import platform
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterable
from enum import Enum, auto
from math import prod
from pathlib import Path

from cffi import FFI
from tqdm import tqdm

ffi = FFI()

class UnaryOps(Enum):
    SUM = auto()
    MAX = auto()
    NEG = auto()
    EXP = auto()
    LOG = auto()
    POW = auto()
    SQRT = auto()
    RSQRT = auto()
    TRANSPOSE = auto()
    RELU = auto()
    ARGMAX = auto()
    WHERE = auto()
    MEAN = auto()
    CLAMP = auto()
    MASKED_FILL = auto()
    PERMUTE = auto()

class BufferOps(Enum):
    CREATE = auto()
    UNIFORM = auto()
    ARANGE = auto()
    FULL = auto()

class BinaryOps(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MATMUL = auto()
    GT = auto() # >
    GTE = auto() # >=
    EQT = auto() # ==
    CMP  = auto()

class CustomOps(Enum):
    INDEX = auto()
    CE_FORWARD = auto()
    CE_BACKWARD = auto()
    PRINT = auto()
    EMBEDDING = auto()


def prod_(x: Iterable) -> int:
    return prod(x) if x else 1

def check_broadcast(x_shape: tuple, y_shape: tuple) -> bool:
    """
    Check if shapes are broadcastable.

    Parameters:
        x_shape (tuple): The x Tensor shape.
        y_shape (tuple): The y Tensor shape.

    Returns:
        bool: True if they are broadcastable.

    Raises:
        AssertionError: If the shapes are not broadcastable.
    """
    for i, j in itertools.zip_longest(reversed(x_shape), reversed(y_shape)):
        if i is not None and j is not None and i != j and i != 1 and j != 1:
            raise AssertionError(f"Cannot broadcast {y_shape} to {x_shape}, the dimensions {i} and {j} dont match")  # noqa: E501

    return True

def find_broadcast_dim(shape1: tuple, shape2: tuple) -> int:
    if len(shape1) != len(shape2):
        raise ValueError("Shapes must have the same number of dimensions")

    for i in range(len(shape1)):
        if (shape1[i] == 1 and shape2[i] != 1) or (shape2[i] == 1 and shape1[i] != 1):
            if shape1[i] == shape2[i] or shape1[i] == 1 or shape2[i] == 1:
                return i

def get_brodcast_tensor(x, y):  # noqa: ANN001, ANN201
    if len(x.shape) > len(y.shape):
        return x, y
    elif len(x.shape) < len(y.shape):
        return y, x
    else:
        for dim_x, dim_y in zip(x.shape, y.shape):
            if dim_x > dim_y:
                return x, y
            elif dim_x < dim_y:
                return y, x

        return x, y

def cal_cat_out_shape(data: tuple[tuple], cat_dim: int) -> tuple:
    transposed = list(zip(*data))
    summed_val = sum(transposed[cat_dim])
    result = list(data[0])
    result[cat_dim] = summed_val
    return tuple(result)

def broadcast_shapes(s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """
    Returns (padded_shape1, padded_shape2, output_shape)
    """
    ndim1, ndim2 = len(s1), len(s2)
    max_ndim = max(ndim1, ndim2)

    p1 = (1,) * (max_ndim - ndim1) + s1
    p2 = (1,) * (max_ndim - ndim2) + s2

    out_batch = []
    for i in range(max_ndim - 2):
        d1, d2 = p1[i], p2[i]
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise ValueError(f"Cannot broadcast batch dims: {s1} and {s2}")
        out_batch.append(max(d1, d2))

    if p1[-1] != p2[-2]:
        raise ValueError(f"Matrix inner dims must match: {s1} and {s2}")

    out_shape = tuple(out_batch) + (p1[-2], p2[-1])
    return p1, p2, out_shape

def get_broadcast_shape(shape1: tuple, shape2: tuple) -> tuple:
    s1 = list(reversed(shape1))
    s2 = list(reversed(shape2))

    max_len = max(len(s1), len(s2))

    final_shape = []

    for i in range(max_len):
        dim1 = s1[i] if i < len(s1) else 1
        dim2 = s2[i] if i < len(s2) else 1

        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            raise ValueError(f"Operands could not be broadcast together with shapes {shape1} and {shape2}")

        final_shape.append(max(dim1, dim2))

    return tuple(reversed(final_shape))

def get_broadcast_strides(src_shape: tuple, dst_shape: tuple) -> tuple:
    """
    Calculates strides for src_shape as if it were broadcast to dst_shape.
    Returns a tuple of strides matching the length of dst_shape.
    """
    base_strides = calculate_stride(src_shape)
    ndim_diff = len(dst_shape) - len(src_shape)
    aligned_src_shape = (1,) * ndim_diff + src_shape
    aligned_src_strides = (0,) * ndim_diff + base_strides
    final_strides = []

    for dim_src, dim_dst, stride_src in zip(aligned_src_shape, dst_shape, aligned_src_strides):
        if dim_src == dim_dst:
            final_strides.append(stride_src)
        elif dim_src == 1:
            final_strides.append(0)
        else:
            raise ValueError(f"Shape mismatch: {src_shape} cannot be broadcast to {dst_shape}")
    return tuple(final_strides)

def calculate_stride(shape: tuple|int) -> tuple:
    if not shape:
        return tuple()

    stride = []
    stride_value = 1
    for dim in reversed(shape):
        stride.append(stride_value)
        stride_value *= dim

    return tuple(reversed(stride))

def resolve_ndim(inp_shape: tuple, grad_shape: tuple) -> int:
    if not check_broadcast(x_shape=inp_shape, y_shape=grad_shape):
        raise AssertionError(f"Cannot reduce grad of shape {grad_shape} to the input shape {inp_shape}")  # noqa: E501

    if inp_shape == grad_shape:
        return 0

    ndim = 0
    dim = len(max(inp_shape, grad_shape)) - 1
    for i, j in itertools.zip_longest(reversed(inp_shape), reversed(grad_shape)):
        dim -= 1
        if i != j:
            ndim += 1

    return ndim

def cal_sum_max_out_shape(ndim: int, dim: int, inp_shape: tuple, keepdim: bool = False) -> tuple:
    if dim == -1:
        return inp_shape if keepdim else ()
    t = list(inp_shape)
    if keepdim:
        t[dim] = 1
    else:
        t.pop(dim)
    return tuple(t)

OSX = platform.system() == "Darwin"
CACHE_DIR = os.path.expanduser("~/Library/Caches/dlgrad" if OSX else "~/.cache/dlgrad")

def fetch(url: str, filename: str) -> None:
    downloads_dir = Path(CACHE_DIR) / "downloads"
    path = downloads_dir / filename

    if not os.path.exists(downloads_dir):
        downloads_dir.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        print(f"Downloading {filename} ...")
        try:
            with urllib.request.urlopen(url) as response:
                total_size = response.length
                chunk_size = 8192

                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                    ascii=True,
                ) as pbar, path.open("wb") as file:

                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        pbar.update(len(chunk))
        except urllib.error.HTTPError as e:
            print(f"Failed to download file. HTTP status: {e.code}")
        except urllib.error.URLError as e:
            print(f"Failed to reach server: {e.reason}")

def unzip(filename: str, save_filename: str) -> None:
    downloads_dir = Path(CACHE_DIR) / "downloads"
    src = downloads_dir / filename
    dst = downloads_dir / save_filename

    if not dst.exists():
        with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
            shutil.copyfileobj(fin, fout)

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'

def get_color(color: str) -> str:
    match color:
        case "green":
            return Colors.GREEN
        case "yellow":
            return Colors.YELLOW
        case "red":
            return Colors.RED
        case "end":
            return Colors.END

def get_broadcast_strides_2(out_shape: tuple, in_shape: tuple, in_stride: tuple) -> tuple:
    """
    Computes effective strides for an input to match the output shape
    according to broadcasting rules.
    """
    ndim_out = len(out_shape)
    ndim_in = len(in_shape)

    # 1. Align Ranks: Pad smaller shape with 1s on the left
    # If out=(4, 3, 2) and in=(3, 2), diff=1.
    # effective_in_shape becomes (1, 3, 2)
    # effective_in_stride becomes (0, stride_0, stride_1)
    diff = ndim_out - ndim_in

    # Pad strides with 0 for the missing leading dimensions
    # (The value doesn't strictly matter since the shape is 1, but 0 is safe)
    padded_in_shape = (1,) * diff + in_shape
    padded_in_stride = (0,) * diff + in_stride

    new_strides = []

    # 2. Compute Broadcasting Strides
    for out_d, in_d, in_s in zip(out_shape, padded_in_shape, padded_in_stride):
        if in_d == 1 and out_d > 1:
            # Broadcasting: This dim is 1 in input but >1 in output.
            # We want to repeat the value, so stride must be 0.
            new_strides.append(0)
        elif in_d == out_d:
            # Dimensions match, keep original stride
            new_strides.append(in_s)
        else:
            raise ValueError(f"Shape mismatch: cannot broadcast {in_shape} to {out_shape}")

    return tuple(new_strides)

def broadcast_shapes_2(s1: tuple, s2: tuple) -> tuple:
    """
    Standard broadcasting logic to find the resulting shape.
    Example: (3, 1) + (3, 4) -> (3, 4)
    """
    # 1. Align ranks
    ndim = max(len(s1), len(s2))

    # Prepend 1s
    s1_filled = (1,) * (ndim - len(s1)) + s1
    s2_filled = (1,) * (ndim - len(s2)) + s2

    out_shape = []
    for d1, d2 in zip(s1_filled, s2_filled):
        if d1 == d2:
            out_shape.append(d1)
        elif d1 == 1:
            out_shape.append(d2)
        elif d2 == 1:
            out_shape.append(d1)
        else:
            raise ValueError(f"Operands could not be broadcast together with shapes {s1} {s2}")

    return tuple(out_shape)

