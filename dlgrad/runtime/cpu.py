import hashlib
import os
import pathlib
import struct
import subprocess
from functools import cache

import _af  # type: ignore
import _allocate  # type: ignore
import _cmp  # type: ignore
import _full  # type: ignore
import _loss  # type: ignore
import _matmul  # type: ignore
import _mnist_loader  # type: ignore
import _transpose  # type: ignore
import _uniform  # type: ignore
import _utils  # type: ignore
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.codegen.cpu import cpu_kernel
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, Scalar
from dlgrad.helpers import CACHE_DIR, BinaryOps, BufferOps, CustomOps, UnaryOps, cal_sum_out_shape, calculate_stride, prod_

CFLAGS = ["-shared", "-fPIC", "-O2", "-march=native"]
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
    def _binary_op(x: Buffer, y: Buffer | Scalar, op: str) -> CDataPtr:
        c_code, cdef = cpu_kernel.arithmetic(x.shape, x.stride, y.shape, y.stride, op)

        key   = CPU._hash_code(c_code)
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
    def add(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        return CPU._binary_op(x, y, op="add")

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.CPU)
    def sub(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        return CPU._binary_op(x, y, op_code=2)

    @staticmethod
    @dispatcher.register(BinaryOps.MUL, Device.CPU)
    def mul(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        return CPU._binary_op(x, y, op_code=1)

    @staticmethod
    @dispatcher.register(BinaryOps.DIV, Device.CPU)
    def div(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        return CPU._binary_op(x, y, op_code=3)

    @staticmethod
    @dispatcher.register(UnaryOps.NEG, Device.CPU)
    def neg(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)
        _utils.lib.neg(x.ptr, out_ptr, x.numel)
        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.TRANSPOSE, Device.CPU)
    def transpose(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)
        _transpose.lib.transpose(x.ptr, out_ptr, x.shape[0], x.shape[1], x.stride, calculate_stride(x.shape[::-1]))
        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.EXP, Device.CPU)
    def exp(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)
        _utils.lib.cexp(x.ptr, out_ptr, x.numel)
        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.SQRT, Device.CPU)
    def sqrt(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)
        _utils.lib.csqrt(x.ptr, out_ptr, x.numel)
        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.LOG, Device.CPU)
    def log(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)
        _utils.lib.clog(x.ptr, out_ptr, x.numel)
        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.POW, Device.CPU)
    def pow(x: Buffer, val: Scalar) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)
        _utils.lib.cpow(x.ptr, out_ptr, val, x.numel)
        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.MATMUL, Device.CPU)
    def matmul(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.shape[0]*y.shape[1])
        _matmul.lib.matmul(x.ptr, y.ptr, out_ptr, x.shape[0], y.shape[1], y.shape[0], y.stride, x.stride)
        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.SUM, Device.CPU)
    def sum(x: Buffer, dim: int) -> CDataPtr:
        num = prod_(cal_sum_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape))
        out_ptr = CPU.calloc(num=num)

        # if x.ndim == 3:
        #     _sum.lib.sum_3d(x.ptr, out_ptr, x.shape, x.stride, x.numel, dim)
        # if x.ndim == 2:
        #     _sum.lib.sum_2d(x.ptr, out_ptr, x.shape, x.stride, x.numel, dim)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.MAX, Device.CPU)
    def max(x: Buffer, dim: int) -> CDataPtr:
        num = prod_(cal_sum_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape))
        out_ptr = CPU.init_with_scalar(num=num, scalar=-999)
        # tmp = CPU.malloc(num=num)
        max_with_1s = CPU.calloc(num=x.numel)

        c_code, cdef = cpu_kernel.max(x.shape, x.stride, x.numel, dim)

        key   = CPU._hash_code(c_code)
        so_fp = pathlib.Path(CACHE_DIR) / f"max_{key}.so"
        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)

        lib = CPU._get_handle(str(so_fp))

        CPU._ensure_sig(cdef)

        lib.max(x.ptr, out_ptr)

        # if x.ndim == 3:
        #     _max.lib.max_3d(x.ptr, out_ptr, tmp, max_with_1s, x.shape, x.stride, x.numel, dim)
        # if x.ndim == 2:
        #     _max.lib.max_2d(x.ptr, out_ptr, tmp, max_with_1s, x.shape, x.stride, x.numel, dim)

        return out_ptr, max_with_1s

    @staticmethod
    @dispatcher.register(UnaryOps.RELU, Device.CPU)
    def relu(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)
        _af.lib.relu(x.ptr, out_ptr, x.numel)
        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.GT, Device.CPU)
    def gt(x: Buffer, y: int | float) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)
        _cmp.lib.gt_with_scalar(x.ptr, out_ptr, y, x.numel)
        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.EQT, Device.CPU)
    def eqt(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)
        _cmp.lib.eqt(x.ptr, y.ptr, out_ptr, x.numel)
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
        _utils.lib.argmax2d(x.ptr, out_ptr, x.shape, axis)
        return out_ptr

    @staticmethod
    @dispatcher.register(CustomOps.CE_FORWARD, Device.CPU)
    def ce_forward(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.shape[0])
        _loss.lib.ce_forward(x.ptr, y.ptr, out_ptr, x.shape[0], x.stride)
        return out_ptr

    @staticmethod
    @dispatcher.register(CustomOps.CE_BACKWARD, Device.CPU)
    def ce_backward(x: Buffer, target: Buffer) -> CDataPtr:
        _loss.lib.ce_backward(x.ptr, target.ptr, x.shape, x.stride)
