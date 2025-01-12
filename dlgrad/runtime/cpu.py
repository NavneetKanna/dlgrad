import _af  # type: ignore
import _allocate  # type: ignore
import _arithmetic  # type: ignore
import _cmp  # type: ignore
import _full  # type: ignore
import _matmul  # type: ignore
import _neg  # type: ignore
import _sum  # type: ignore
import _uniform  # type: ignore
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, DType, Scalar
from dlgrad.helpers import BinaryOps, BufferOps, UnaryOps, prod_


# TODO: Calling ffi.gc() twice one after the other leads to error, find alternative
# TODO: Should numel be sent as arg or be calculated here ?
class CPU:
    """
    Main CPU runtime class which handles the logic of calling the compiled C source files.

    This class uses CFFI (C Foreign Function Interface) to interact with C code.
    """
    ffi = FFI()

    @staticmethod
    def allocate(num: int, initialize: bool = False) -> CDataPtr:
        if not initialize:
            return CPU.ffi.gc(_allocate.lib.uninitialized_memory(num), _allocate.lib.free_ptr)
        return CPU.ffi.gc(_allocate.lib.initialized_memory(num), _allocate.lib.free_ptr)

    @staticmethod
    @dispatcher.register(BufferOps.CREATE, Device.CPU)
    def create_buffer_from_scalar(x: Scalar) -> CDataPtr:
        return CPU.ffi.new(f"{DType.get_c_dtype(x)} [1]", [x])

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
        out_ptr = CPU.allocate(num=x.numel)

        match x.ndim:
            case 3:
                _arithmetic.lib.op_3d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 0)
            case 2:
                _arithmetic.lib.op_2d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 0)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.CPU)
    def sub(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.allocate(num=x.numel)

        match x.ndim:
            case 3:
                _arithmetic.lib.op_3d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 2)
            case 2:
                _arithmetic.lib.op_2d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 2)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.MUL, Device.CPU)
    def mul(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.allocate(num=x.numel)

        match x.ndim:
            case 3:
                _arithmetic.lib.op_3d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 1)
            case 2:
                _arithmetic.lib.op_2d(x.ptr, y.ptr, out_ptr, x.shape, x.stride, y.shape, y.stride, 1)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.NEG, Device.CPU)
    def neg(x: Buffer) -> CDataPtr:
        out_ptr = CPU.allocate(num=x.numel)

        _neg.lib.neg(x.ptr, out_ptr, x.numel)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.MATMUL, Device.CPU)
    def matmul(x: Buffer, y: Buffer) -> CDataPtr:
        out_ptr = CPU.allocate(num=x.shape[0]*y.shape[1])

        _matmul.lib.matmul(x.ptr, y.ptr, out_ptr, x.shape[0], y.shape[1], y.shape[0], y.stride, x.stride)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.SUM, Device.CPU)
    def sum(x: Buffer, dim: int | None, numel: int) -> CDataPtr:
        if x.ndim == 3:
            arr = _sum.lib.sum_3d(x.ptr, x.shape, x.stride, numel, dim)
        if x.ndim == 2:
            arr = _sum.lib.sum_2d(x.ptr, x.shape, x.stride, numel, dim)

        return CPU.ffi.gc(arr, _sum.lib.free_sum)

    @staticmethod
    @dispatcher.register(UnaryOps.RELU, Device.CPU)
    def relu(x: Buffer, numel: int) -> CDataPtr:
        arr = _af.lib.relu(x.ptr, numel)
        return CPU.ffi.gc(arr, _af.lib.free_af)

    @staticmethod
    @dispatcher.register(BinaryOps.GT, Device.CPU)
    def gt(x: Buffer, y: int | float) -> CDataPtr:
        if isinstance(y, int):
            y = float(y)

        arr = _cmp.lib.gt_with_scalar(x.ptr, y, x.numel)

        return CPU.ffi.gc(arr, _cmp.lib.free_cmp)

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
