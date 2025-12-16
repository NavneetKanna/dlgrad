import hashlib
import os
import pathlib
import shutil
import struct
import subprocess
import sys
from functools import cache

from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.codegen import cpu_kernel
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, Scalar
from dlgrad.helpers import (
   CACHE_DIR,
   BinaryOps,
   BufferOps,
   CustomOps,
   UnaryOps,
   cal_sum_max_out_shape,
   calculate_stride,
   prod_,
)

CFLAGS = ["-shared", "-fPIC", "-O3", "-march=native", "-ffast-math", "-Werror"]
if shutil.which('clang'):
   COMPILER = "clang"
elif shutil.which('gcc'):
    COMPILER = "gcc"
else:
    print("Clang or GCC not found")
    sys.exit(1)

EXTRA_LIBS = []
if sys.platform == "linux":
    EXTRA_LIBS = ["-lm", "-lmvec"]

class CPU:
    """
    Main CPU runtime class which handles the logic of calling the compiled C source files.

    This class uses CFFI (C Foreign Function Interface) to interact with C code.
    """
    ffi = FFI()

    @staticmethod
    def _build_shared_object(source: str, so_path: pathlib.Path) -> None:
        c_path = so_path.with_suffix(".c")
        c_path.write_text(source)

        cmd = [COMPILER, *CFLAGS, "-o", str(so_path), str(c_path), *EXTRA_LIBS]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{res.stderr}")

    @staticmethod
    @cache
    def _hash_code(code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()

    @staticmethod
    @cache
    def _get_handle(so_path: str):  # noqa: ANN205
        return CPU.ffi.dlopen(so_path)

    @staticmethod
    @cache
    def _ensure_sig(cdef: str) -> bool:
        CPU.ffi.cdef(cdef)
        return True

    # TODO: Cache struct.calcsize('f')
    @staticmethod
    def malloc(num: int, size: int = struct.calcsize('f')) -> CDataPtr:
        c_code, cdef = cpu_kernel.uninitialized_memory()
        c_code2, cdef2 = cpu_kernel.free_ptr()

        key = CPU._hash_code(c_code)
        key2 = CPU._hash_code(c_code2)
        so_fp = pathlib.Path(CACHE_DIR) / f"unintialized_memory_{key}.so"
        so_fp2 = pathlib.Path(CACHE_DIR) / f"free_{key2}.so"

        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)
        if not os.path.exists(so_fp2):
            CPU._build_shared_object(c_code2, so_fp2)

        lib = CPU._get_handle(str(so_fp))
        lib2 = CPU._get_handle(str(so_fp2))

        CPU._ensure_sig(cdef)
        CPU._ensure_sig(cdef2)

        ptr = CPU.ffi.gc(lib.uninitialized_memory(num*size), lib2.free_ptr)

        if ptr == CPU.ffi.NULL:
            raise MemoryError(f"Failed to allocate {num * size} bytes of memory")

        return ptr

    @staticmethod
    def calloc(num: int, size: int = struct.calcsize('f')) -> CDataPtr:
        c_code, cdef = cpu_kernel.initialized_memory()
        c_code2, cdef2 = cpu_kernel.free_ptr()

        key = CPU._hash_code(c_code)
        key2 = CPU._hash_code(c_code2)
        so_fp = pathlib.Path(CACHE_DIR) / f"initialized_memory_{key}.so"
        so_fp2 = pathlib.Path(CACHE_DIR) / f"free_{key2}.so"

        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)
        if not os.path.exists(so_fp2):
            CPU._build_shared_object(c_code2, so_fp2)

        lib = CPU._get_handle(str(so_fp))
        lib2 = CPU._get_handle(str(so_fp2))

        CPU._ensure_sig(cdef)
        CPU._ensure_sig(cdef2)

        ptr = CPU.ffi.gc(lib.initialized_memory(num, size), lib2.free_ptr)

        if ptr == CPU.ffi.NULL:
            raise MemoryError(f"Failed to allocate {num * size} bytes of memory")
        return ptr

    @staticmethod
    def mnist_loader(images: bool, path: str, magic_number: int) -> CDataPtr:
        c_code, cdef = cpu_kernel.mnist_loader()
        c_code2, cdef2 = cpu_kernel.free_ptr()

        key = CPU._hash_code(c_code)
        key2 = CPU._hash_code(c_code2)
        so_fp = pathlib.Path(CACHE_DIR) / f"mnist_loader_{key}.so"
        so_fp2 = pathlib.Path(CACHE_DIR) / f"free_{key2}.so"

        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)
        if not os.path.exists(so_fp2):
            CPU._build_shared_object(c_code2, so_fp2)

        lib = CPU._get_handle(str(so_fp))
        lib2 = CPU._get_handle(str(so_fp2))

        CPU._ensure_sig(cdef)
        CPU._ensure_sig(cdef2)

        if images:
            ptr = CPU.ffi.gc(lib.mnist_images_loader(path.encode('ascii'), magic_number), lib2.free_ptr)  # noqa: E501
        else:
            ptr = CPU.ffi.gc(lib.mnist_labels_loader(path.encode('ascii'), magic_number), lib2.free_ptr)  # noqa: E501

        if ptr == CPU.ffi.NULL:
            raise MemoryError("Error when loading MNIST data")
        return ptr

    @staticmethod
    def init_with_scalar(num: int, scalar: int, size: int = struct.calcsize('f')) -> CDataPtr:
        c_code, cdef = cpu_kernel.init_with_scalar()
        c_code2, cdef2 = cpu_kernel.free_ptr()

        key = CPU._hash_code(c_code)
        key2 = CPU._hash_code(c_code2)
        so_fp = pathlib.Path(CACHE_DIR) / f"init_with_scalar_{key}.so"
        so_fp2 = pathlib.Path(CACHE_DIR) / f"free_{key2}.so"

        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)
        if not os.path.exists(so_fp2):
            CPU._build_shared_object(c_code2, so_fp2)

        lib = CPU._get_handle(str(so_fp))
        lib2 = CPU._get_handle(str(so_fp2))

        CPU._ensure_sig(cdef)
        CPU._ensure_sig(cdef2)

        ptr = CPU.ffi.gc(lib.init_with_scalar(num*size, num, scalar), lib2.free_ptr)

        if ptr == CPU.ffi.NULL:
            raise MemoryError(f"Failed to allocate {num * size} bytes of memory")
        return ptr

    @staticmethod
    @dispatcher.register(BufferOps.UNIFORM, Device.CPU)
    def uniform(shape: tuple, low: float, high: float) -> CDataPtr:
        out_ptr = CPU.malloc(num=prod_(shape))

        c_code, cdef = cpu_kernel.uniform(prod_(shape), low, high)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"uniform_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        status = lib.uniform(out_ptr)

        if status == -1:
            raise MemoryError("Failed to create random values")

        return out_ptr

    @staticmethod
    @dispatcher.register(BufferOps.ARANGE, Device.CPU)
    def arange(shape: tuple) -> CDataPtr:
        out_ptr = CPU.malloc(num=prod_(shape))

        c_code, cdef = cpu_kernel.arange(prod_(shape))

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"arange_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.arange(out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(BufferOps.FULL, Device.CPU)
    def full(shape: tuple, fill_value: Scalar) -> CDataPtr:
        out_ptr = CPU.malloc(num=prod_(shape))

        c_code, cdef = cpu_kernel.full(prod_(shape), fill_value)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"full_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.full(out_ptr)

        return out_ptr

    @staticmethod
    def _binary_op(x: Buffer, y: Buffer, op: str) -> CDataPtr:
        c_code, cdef = cpu_kernel.arithmetic(x.shape, x.stride, y.shape, y.stride, op)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"{op}_{key}.so"

        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        func = getattr(lib, op)
        outptr = CPU.malloc(num=x.numel)
        func(x.ptr, y.ptr, outptr)

        return outptr

    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.CPU)
    def add(x: Buffer, y: Buffer) -> CDataPtr:
        return CPU._binary_op(x, y, op="add")

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.CPU)
    def sub(x: Buffer, y: Buffer) -> CDataPtr:
        return CPU._binary_op(x, y, op="sub")

    @staticmethod
    @dispatcher.register(BinaryOps.MUL, Device.CPU)
    def mul(x: Buffer, y: Buffer) -> CDataPtr:
        return CPU._binary_op(x, y, op="mul")

    @staticmethod
    @dispatcher.register(BinaryOps.DIV, Device.CPU)
    def div(x: Buffer, y: Buffer) -> CDataPtr:
        return CPU._binary_op(x, y, op="divv")

    @staticmethod
    @dispatcher.register(UnaryOps.NEG, Device.CPU)
    def neg(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.utils(x.numel, "neg")

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"neg_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.c_neg(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.TRANSPOSE, Device.CPU)
    def transpose(x: Buffer, out_stride: tuple, dim0: int, dim1: int) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        if x.ndim == 3 and ((dim0 == 0 and dim1 == 1) or (dim0 == 1 and dim1 == 0)):
            c_code, cdef = cpu_kernel.transpose_3d_01(x.shape, (x.shape[1], x.shape[0], x.shape[2]), x.stride, out_stride, x.numel)
        elif x.ndim == 3 and ((dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1)):
            c_code, cdef = cpu_kernel.transpose_3d_12(x.shape, (x.shape[1], x.shape[0], x.shape[2]), x.stride, out_stride, x.numel)
        else:
            c_code, cdef = cpu_kernel.transpose_2d(x.shape, x.stride, out_stride, x.numel)

        key   = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"transpose_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.transpose(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.EXP, Device.CPU)
    def exp(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.utils(x.numel, "exp")

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"exp_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.c_exp(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.SQRT, Device.CPU)
    def sqrt(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.utils(x.numel, "sqrt")

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"sqrt_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.c_sqrt(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.CLAMP, Device.CPU)
    def clamp(x: Buffer, min: int | None, max: int | None) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.clamp(x.numel, min, max)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"clamp_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.clamp(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.LOG, Device.CPU)
    def log(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.utils(x.numel, "log")

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"log_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.c_log(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.POW, Device.CPU)
    def pow(x: Buffer, val: Scalar) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.utils(x.numel, "pow", int(val))

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"pow_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.c_pow(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.MATMUL, Device.CPU)
    def matmul(x: Buffer, y: Buffer) -> CDataPtr:
        if x.ndim == 3:
            if x.shape[0] != 1:
                out_shape = x.shape[0]*x.shape[1]*y.shape[2]
                out_stride = calculate_stride((x.shape[0], x.shape[1], y.shape[2]))
            else:
                out_shape = y.shape[0]*x.shape[1]*y.shape[2]
                out_stride = calculate_stride((y.shape[0], x.shape[1], y.shape[2]))
            out_ptr = CPU.init_with_scalar(num=out_shape, scalar=0)
            broadcast_x, broadcast_y = False, False
            if x.shape[0] == 1:
                broadcast_x = True
            if y.shape[0] == 1:
                broadcast_y = True
            c_code, cdef = cpu_kernel.matmul_3d(x.shape, y.shape, x.stride, y.stride, out_stride, broadcast_x, broadcast_y)
        else:
            out_ptr = CPU.init_with_scalar(num=x.shape[0]*y.shape[1], scalar=0)
            c_code, cdef = cpu_kernel.matmul_2d(x.shape, y.shape, x.stride, y.stride)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"matmul_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.matmul(x.ptr, y.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.SUM, Device.CPU)
    def sum(x: Buffer, dim: int) -> CDataPtr:
        out_shape = cal_sum_max_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape)
        out_stride = calculate_stride(out_shape)
        num = prod_(out_shape)
        out_ptr = CPU.calloc(num=num)

        if x.ndim == 4:
            c_code, cdef = cpu_kernel.reduce_4d(x.shape, x.stride, out_stride, x.numel, dim, "sum")
        elif x.ndim == 3:
            c_code, cdef = cpu_kernel.reduce_3d(x.shape, x.stride, out_stride, x.numel, dim, "sum")
        elif x.ndim == 2:
            c_code, cdef = cpu_kernel.reduce_2d(x.shape, x.stride, out_stride, x.numel, dim, "sum")
        elif x.ndim == 1 and dim == -1:
            c_code, cdef = cpu_kernel.reduce(x.numel, "sum")

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"sum_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.sum(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.MEAN, Device.CPU)
    def mean(x: Buffer, dim: int) -> CDataPtr:
        num = prod_(cal_sum_max_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape))
        out_ptr = CPU.calloc(num=num)

        c_code, cdef = cpu_kernel.mean(x.shape, x.stride, x.numel, dim, num)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"mean_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.mean(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.MAX, Device.CPU)
    def max(x: Buffer, dim: int, out: Buffer = None) -> CDataPtr:
        out_shape = cal_sum_max_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape)
        out_stride = calculate_stride(out_shape)
        num = prod_(out_shape)
        out_ptr = CPU.init_with_scalar(num=num, scalar=-999)

        if x.ndim == 4:
            c_code, cdef = cpu_kernel.reduce_4d(x.shape, x.stride, out_stride, x.numel, dim, "max")
        elif x.ndim == 3:
            c_code, cdef = cpu_kernel.reduce_3d(x.shape, x.stride, out_stride, x.numel, dim, "max")
        elif x.ndim == 2:
            c_code, cdef = cpu_kernel.reduce_2d(x.shape, x.stride, out_stride, x.numel, dim, "max")
        elif x.ndim == 1 and dim == -1:
            c_code, cdef = cpu_kernel.reduce(x.numel, "max")

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"max_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.max(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.RELU, Device.CPU)
    def relu(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.relu(x.numel)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"relu_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.relu(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.GT, Device.CPU)
    def gt(x: Buffer, y: int | float) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.gt(x.numel, y)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"gt_with_scalar_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.gt_with_scalar(x.ptr, out_ptr)

        return out_ptr

    # TODO: Add to docs once testing
    @staticmethod
    @dispatcher.register(BinaryOps.CMP, Device.CPU)
    def cmp(x: Buffer, y: Buffer | int | float, out_shape: tuple,  mode: str) -> CDataPtr:
        out_ptr = CPU.malloc(num=prod_(out_shape))

        match mode:
            case "<=":
                x_stride = tuple([0 if i== 1 else 1 for i in x.shape])
                y_stride = tuple([0 if i== 1 else 1 for i in y.shape])
                c_code, cdef = cpu_kernel.cmp_2d(mode, out_shape, calculate_stride(out_shape), x.shape, x_stride, y_stride)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"cmp_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.cmp(x.ptr, y.ptr,  out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.WHERE, Device.CPU)
    def where(x: Buffer, inp: Buffer, other: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.where(x.numel, inp=True if inp.ndim == 0 else False, other=True if other.ndim == 0 else False)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"where_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.where(x.ptr, out_ptr, inp.ptr, other.ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.EQT, Device.CPU)
    def eqt(x: Buffer, y: Buffer) -> CDataPtr:
        # NOTE: Assumes x.numel > y.numel
        # TODO: Add check that the numel should be the same
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.eqt(x.numel, True if y.ndim == 0 else False, x.shape, x.stride, y.shape, y.stride, x.ndim)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"eqt_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.eqt(x.ptr, y.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.ARGMAX, Device.CPU)
    def argmax(x: Buffer, dim: int) -> CDataPtr:
        if dim==0:
            n = x.shape[1]
        elif dim==1:
            n = x.shape[0]
        else:
            n = 1
        out_ptr = CPU.malloc(num=n)

        c_code, cdef = cpu_kernel.argmax(x.shape, dim)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"argmax_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.argmax2d(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(CustomOps.CE_FORWARD, Device.CPU)
    def ce_forward(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.shape[0])

        c_code, cdef = cpu_kernel.ce_forward(x.shape[0], x.stride)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"ce_forward_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.ce_forward(x.ptr, y.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(CustomOps.CE_BACKWARD, Device.CPU)
    def ce_backward(x: Buffer, target: Buffer) -> CDataPtr:
        c_code, cdef = cpu_kernel.ce_backward(x.shape, x.stride)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"ce_backward_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.ce_backward(x.ptr, target.ptr)

    @staticmethod
    @dispatcher.register(CustomOps.PRINT, Device.CPU)
    def show(x: Buffer) -> None:
        if x.ndim == 0:
            c_code, cdef = cpu_kernel.print_0d_tensor()
        elif x.ndim == 1:
            c_code, cdef = cpu_kernel.print_1d_tensor(x.shape, x.stride, x.numel)
        elif x.ndim == 2:
            c_code, cdef = cpu_kernel.print_2d_tensor(x.shape, x.stride, x.numel)
        elif x.ndim == 3:
            c_code, cdef = cpu_kernel.print_3d_tensor(x.shape, x.stride, x.numel)
        elif x.ndim == 4:
            c_code, cdef = cpu_kernel.print_4d_tensor(x.shape, x.stride, x.numel)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"print_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.print_tensor(x.ptr)

    @staticmethod
    @dispatcher.register(CustomOps.EMBEDDING, Device.CPU)
    def embedding(x: Buffer, idx: Buffer, out_numel: int, backward: bool = False, upstream_grad: Buffer = None) -> CDataPtr:
        if backward:
            out_ptr = CPU.init_with_scalar(num=x.numel, scalar=0)
            c_code, cdef = cpu_kernel.embedding_backward(idx.numel, x.shape[-1])
            key = CPU._hash_code(c_code)
            so_fp = pathlib.Path(CACHE_DIR) / f"embedding_backward_{key}.so"
            if not os.path.exists(so_fp):
                CPU._build_shared_object(c_code, so_fp)

            lib = CPU._get_handle(str(so_fp))

            CPU._ensure_sig(cdef)

            lib.embedding_backward(out_ptr, upstream_grad.ptr, idx.ptr)

            return out_ptr

        out_ptr = CPU.malloc(num=out_numel)

        c_code, cdef = cpu_kernel.embedding(idx.numel, x.shape[-1])

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"embedding_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.embedding(x.ptr, idx.ptr, out_ptr)

        return out_ptr
