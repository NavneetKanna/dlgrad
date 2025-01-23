import struct

import _af  # type: ignore
import _allocate  # type: ignore
import _arithmetic  # type: ignore
import _cmp  # type: ignore
import _full  # type: ignore
import _index  # type: ignore
import _matmul  # type: ignore
import _max  # type: ignore
import _sum  # type: ignore
import _uniform  # type: ignore
import _utils  # type: ignore
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, DType, Scalar
from dlgrad.helpers import BinaryOps, BufferOps, CustomOps, UnaryOps, cal_sum_out_shape, prod_


# TODO: Calling ffi.gc() twice one after the other leads to error, find alternative
class CPU:
    """
    Main CPU runtime class which handles the logic of calling the compiled C source files.

    This class uses CFFI (C Foreign Function Interface) to interact with C code.
    """
    ffi = FFI()

    @staticmethod
    def malloc(num: int, size: int = struct.calcsize('f')) -> CDataPtr:
        ptr = CPU.ffi.gc(_allocate.lib.uninitialized_memory(num*size), _allocate.lib.free_ptr)
        if ptr == CPU.ffi.NULL:
            raise MemoryError(f"Unable to allocate requested memory of size {num*size} bytes")
        return ptr

    @staticmethod
    def calloc(num: int, size: int = struct.calcsize('f')) -> CDataPtr:
        ptr = CPU.ffi.gc(_allocate.lib.initialized_memory(num, size), _allocate.lib.free_ptr)
        if ptr == CPU.ffi.NULL:
            raise MemoryError(f"Unable to allocate requested memory of size {num*size} bytes")
        return ptr

    @staticmethod
    def init_with_scalar(num: int, scalar: int, size: int = struct.calcsize('f')) ->CDataPtr:
        ptr = CPU.ffi.gc(_allocate.lib.init_with_scalar(num*size, num, scalar), _allocate.lib.free_ptr)
        if ptr == CPU.ffi.NULL:
            raise MemoryError(f"Unable to allocate requested memory of size {num*size} bytes")
        return ptr

    @staticmethod
    @dispatcher.register(BufferOps.CREATE, Device.CPU)
    def create_buffer_from_scalar(x: Scalar) -> CDataPtr:
        return CPU.ffi.new(f"{DType.get_c_dtype(x)} [1]", [x]) # TODO: Is this the right thing to do ?

    @staticmethod
    @dispatcher.register(BufferOps.UNIFORM, Device.CPU)
    def uniform(shape: tuple, low: float, high: float) -> CDataPtr:
        numel = prod_(shape)
        arr = _uniform.lib.uniform(numel, low, high)

        return CPU.ffi.gc(arr, _uniform.lib.free_uniform)

    @staticmethod
    @dispatcher.register(BufferOps.FULL, Device.CPU)
    def full(shape: tuple, fill_value: Scalar) -> CDataPtr:
        arr = _full.lib.full(prod_(shape), fill_value)

        return CPU.ffi.gc(arr, _full.lib.free_full)

    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.CPU)
    def add(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        if y.ndim == 1:
            _arithmetic.lib.add_with_1d(x.ptr, y.ptr, out_ptr, x.numel, y.numel, 0)
            return out_ptr

        match x.ndim:
            case 3:
                _arithmetic.lib.op_3d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 0)
            case 2:
                _arithmetic.lib.op_2d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 0)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.CPU)
    def sub(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        if y.ndim == 1:
            _arithmetic.lib.add_with_1d(x.ptr, y.ptr, out_ptr, x.numel, y.numel, 2)
            return out_ptr

        match x.ndim:
            case 3:
                _arithmetic.lib.op_3d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 2)
            case 2:
                _arithmetic.lib.op_2d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 2)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.MUL, Device.CPU)
    def mul(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        if y.ndim == 1:
            _arithmetic.lib.add_with_1d(x.ptr, y.ptr, out_ptr, x.numel, y.numel, 1)
            return out_ptr

        match x.ndim:
            case 3:
                _arithmetic.lib.op_3d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 1)
            case 2:
                _arithmetic.lib.op_2d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 1)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.NEG, Device.CPU)
    def neg(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        _utils.lib.neg(x.ptr, out_ptr, x.numel)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.EXP, Device.CPU)
    def exp(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        _utils.lib.cexp(x.ptr, out_ptr, x.numel)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.LOG, Device.CPU)
    def log(x: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        _utils.lib.clog(x.ptr, out_ptr, x.numel)

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

        if x.ndim == 3:
            _sum.lib.sum_3d(x.ptr, out_ptr, x.shape, x.stride, x.numel, dim)
        if x.ndim == 2:
            _sum.lib.sum_2d(x.ptr, out_ptr, x.shape, x.stride, x.numel, dim)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.MAX, Device.CPU)
    def max(x: Buffer, dim: int) -> CDataPtr:
        num = prod_(cal_sum_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape))
        out_ptr = CPU.init_with_scalar(num=num, scalar=-999)
        tmp = CPU.malloc(num=num)
        max_with_1s = CPU.calloc(num=x.numel)

        if x.ndim == 3:
            _max.lib.max_3d(x.ptr, out_ptr, tmp, max_with_1s, x.shape, x.stride, x.numel, dim)
        if x.ndim == 2:
            _max.lib.max_2d(x.ptr, out_ptr, tmp, max_with_1s, x.shape, x.stride, x.numel, dim)

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

        if isinstance(y, int):
            y = float(y)

        _cmp.lib.gt_with_scalar(x.ptr, out_ptr, y, x.numel)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.EQT, Device.CPU)
    def eqt(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.malloc(num=x.numel)

        if isinstance(y, int):
            y = float(y)

        _cmp.lib.eqt(x.ptr, y, out_ptr, x.numel)

        return out_ptr

    @staticmethod
    @dispatcher.register(CustomOps.INDEX, Device.CPU)
    # only for nll loss
    def index(x: Buffer, idx):  # noqa: ANN001, ANN205
        out_ptr = CPU.malloc(num=len(idx[0]))

        _index.lib.indexing(x.ptr, out_ptr, x.shape, x.stride, idx[1])

        return out_ptr


"""

    (2, 3, 4) + (1, 3, 4)
    (2, 3, 4) + (2, 1, 4)
    (2, 3, 4) + (2, 3, 1)
    (2, 3, 4) + (1, 1, 4)
    (2, 3, 4) + (1, 3, 1)
    (2, 3, 4) + (2, 1, 1)
    (2, 3, 4) + (1, 1, 1)
    (m, n, p) + (1, n, p)
    (m, n, p) + (m, 1, p)
    (m, n, p) + (m, n, 1)
    (m, n, p) + (1, 1, p)
    (m, n, p) + (1, n, 1)
    (m, n, p) + (m, 1, 1)
    (m, n, p) + (1, 1, 1)



    (2, 3) + (1, 3)
    (2, 3) + (2, 1)
    (2, 3) + (1, 1)
    (m, n) + (1, n)
    (m, n) + (m, 1)
    (m, n) + (1, 1)


    (a, b, c, d) + (1, b, c, d)
    (a, b, c, d) + (a, 1, c, d)
    (a, b, c, d) + (a, b, 1, d)
    (a, b, c, d) + (a, b, c, 1)
    (a, b, c, d) + (1, 1, c, d)
    (a, b, c, d) + (1, b, 1, d)
    (a, b, c, d) + (1, b, c, 1)
    (a, b, c, d) + (a, 1, 1, d)
    (a, b, c, d) + (a, 1, c, 1)
    (a, b, c, d) + (a, b, 1, 1)
    (a, b, c, d) + (1, 1, 1, d)
    (a, b, c, d) + (1, 1, c, 1)
    (a, b, c, d) + (1, b, 1, 1)
    (a, b, c, d) + (a, 1, 1, 1)
    (a, b, c, d) + (1, 1, 1, 1)
"""
