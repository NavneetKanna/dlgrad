import hashlib
import os
import pathlib
import struct
import subprocess
from functools import cache

import _allocate  # type: ignore
import _full  # type: ignore
import _mnist_loader  # type: ignore
import _uniform  # type: ignore
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.codegen.cpu import cpu_kernel
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, Scalar
from dlgrad.helpers import CACHE_DIR, BinaryOps, BufferOps, CustomOps, UnaryOps, cal_sum_max_out_shape, prod_

CFLAGS = ["-shared", "-fPIC", "-O2", "-march=native", "-ffast-math"]
COMPILER = "clang"

class CPU:
    """
    Main CPU runtime class which handles the logic of calling the compiled C source files.

    This class uses CFFI (C Foreign Function Interface) to interact with C code.
    """
    ffi = FFI()

    # TODO: Cache struct.calcsize('f')
    @staticmethod
    def malloc(num: int, size: int = struct.calcsize('f')) -> CDataPtr:
        ptr = CPU.ffi.gc(_allocate.lib.uninitialized_memory(num*size), _allocate.lib.free_ptr)
        if ptr == CPU.ffi.NULL:
            raise MemoryError(f"Failed to allocate {num * size} bytes of memory")
        return ptr

    @staticmethod
    def calloc(num: int, size: int = struct.calcsize('f')) -> CDataPtr:
        ptr = CPU.ffi.gc(_allocate.lib.initialized_memory(num, size), _allocate.lib.free_ptr)
        if ptr == CPU.ffi.NULL:
            raise MemoryError(f"Failed to allocate {num * size} bytes of memory")
        return ptr

    @staticmethod
    def mnist_loader(images: bool, path: str, magic_number: int) -> CDataPtr:
        if images:
            ptr = CPU.ffi.gc(_mnist_loader.lib.mnist_images_loader(path.encode('ascii'), magic_number), _allocate.lib.free_ptr)  # noqa: E501
        else:
            ptr = CPU.ffi.gc(_mnist_loader.lib.mnist_labels_loader(path.encode('ascii'), magic_number), _allocate.lib.free_ptr)  # noqa: E501

        if ptr == CPU.ffi.NULL:
            raise MemoryError("Error when loading MNIST data")
        return ptr

    @staticmethod
    def init_with_scalar(num: int, scalar: int, size: int = struct.calcsize('f')) -> CDataPtr:
        ptr = CPU.ffi.gc(_allocate.lib.init_with_scalar(num*size, num, scalar), _allocate.lib.free_ptr)
        if ptr == CPU.ffi.NULL:
            raise MemoryError(f"Failed to allocate {num * size} bytes of memory")
        return ptr

    @staticmethod
    @dispatcher.register(BufferOps.UNIFORM, Device.CPU)
    def uniform(shape: tuple, low: float, high: float) -> CDataPtr:
        out_ptr = CPU.malloc(num=prod_(shape))

        status = _uniform.lib.uniform(out_ptr, prod_(shape), low, high)
        if status == -1:
            raise MemoryError("Failed to create random values")

        return out_ptr

    @staticmethod
    @dispatcher.register(BufferOps.FULL, Device.CPU)
    def full(shape: tuple, fill_value: Scalar) -> CDataPtr:
        out_ptr = CPU.malloc(num=prod_(shape))

        _full.lib.full(out_ptr, prod_(shape), fill_value)

        return out_ptr

    @staticmethod
    def _build_shared_object(source: str, so_path: pathlib.Path) -> None:
        c_path = so_path.with_suffix(".c")
        c_path.write_text(source)

        cmd = [COMPILER, *CFLAGS, "-o", str(so_path), str(c_path)]
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

    # TODO: Cache malloc/out_ptr, reuse it, this should speed up
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

        lib.neg(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.TRANSPOSE, Device.CPU)
    def transpose(x: Buffer, out_shape: tuple, out_stride: tuple, axes: tuple) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.transpose(x.shape, x.stride, out_stride, x.numel, axes)

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

        lib.cexp(x.ptr, out_ptr)

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

        lib.csqrt(x.ptr, out_ptr)

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

        lib.clog(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.POW, Device.CPU)
    def pow(x: Buffer, val: Scalar) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.utils(x.numel, "pow", val)

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
        out_ptr = CPU.malloc(num=x.shape[0]*y.shape[1])

        c_code, cdef = cpu_kernel.matmul(x.shape, y.shape, x.stride, y.stride)

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
        num = prod_(cal_sum_max_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape))
        out_ptr = CPU.calloc(num=num)

        c_code, cdef = cpu_kernel.sum(x.shape, x.stride, x.numel, dim)

        key = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"sum_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.sum(x.ptr, out_ptr)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.MAX, Device.CPU)
    def max(x: Buffer, dim: int, backward: bool = False, out: Buffer = None) -> CDataPtr:
        if backward:
            max_with_1s = CPU.init_with_scalar(num=x.numel, scalar=0)
            c_code, cdef = cpu_kernel.max_backward(x.shape, x.stride, x.numel, dim)
            key = CPU._hash_code(c_code)
            so_fp = pathlib.Path(CACHE_DIR) / f"max_backward_{key}.so"
            if not os.path.exists(so_fp):
                CPU._build_shared_object(c_code, so_fp)

            lib = CPU._get_handle(str(so_fp))

            CPU._ensure_sig(cdef)

            lib.max_backward(x.ptr, out.ptr, max_with_1s)

            return max_with_1s

        num = prod_(cal_sum_max_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape))
        out_ptr = CPU.init_with_scalar(num=num, scalar=-999)

        c_code, cdef = cpu_kernel.max(x.shape, x.stride, x.numel, dim)

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

    @staticmethod
    @dispatcher.register(BinaryOps.EQT, Device.CPU)
    def eqt(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        c_code, cdef = cpu_kernel.eqt(x.numel)

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
    def argmax(x: Buffer, axis: int) -> CDataPtr:
        if axis==0:
            n = x.shape[1]
        elif axis==1:
            n = x.shape[0]
        else:
            n = 1
        out_ptr = CPU.malloc(num=n)

        c_code, cdef = cpu_kernel.argmax(x.shape, axis)

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
