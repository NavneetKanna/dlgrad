from collections.abc import Callable

import _add  # type: ignore
import _af  # type: ignore
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


def _get_add_func(x: Buffer, y: Buffer) -> Callable:
    sh = tuple([s1 if s1 == s2 else 1 for s1, s2 in zip(x.shape, y.shape)])
    shape_map = {
        (1, x.shape[1], x.shape[2]): lambda a, b: _add.lib.add_3d_with_2d(a, b, x.numel, y.numel),
        (x.shape[0], 1, x.shape[2]): lambda a, b: _add.lib.add_with_dim1_with_dim0(a, b, x.numel, y.numel, x.shape[1]*x.shape[2], x.shape[2]),  # noqa: E501
        (x.shape[0], x.shape[1], 1): lambda a, b: _add.lib.add_with_dim0(a, b, x.numel, y.numel, x.shape[2]),
        (1, 1, x.shape[2]): lambda a, b: _add.lib.add_with_dim1(a, b, x.numel, x.shape[2]),
        (1, x.shape[1], 1): lambda a, b: _add.lib.add_with_dim0(a, b, x.numel, y.numel, x.shape[2]),
        (x.shape[0], 1, 1): lambda a, b: _add.lib.add_with_dim0(a, b, x.numel, y.numel, x.shape[1]*x.shape[2]),  # noqa: E501
        (1, 1, 1): lambda a, b: _add.lib.add_with_scalar(a, b, x.numel),
        (x.shape): lambda a, b: _add.lib.add(a, b, x.numel),
        (1, x.shape[1]): lambda a, b: _add.lib.add_with_dim1(a, b, x.numel, x.shape[1]),
        (x.shape[0], 1): lambda a, b: _add.lib.add_with_dim0(a, b, x.numel, y.numel, x.shape[1]),
        (1, 1): lambda a, b: _add.lib.add_with_scalar(a, b, x.numel),
        # (x.shape): lambda a, b: _add.lib.add(a, b, x.numel),
    }

    exit()
    return shape_map[sh]

# TODO: Calling ffi.gc() twice one after the other leads to error, find alternative
class CPU:
    """
    Main CPU runtime class which handles the logic of calling the compiled C source files.

    This class uses CFFI (C Foreign Function Interface) to interact with C code.
    """
    ffi = FFI()

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
    def add(x: Buffer, y: Buffer) -> CDataPtr:  # noqa: C901
        arr = _get_add_func(x=x, y=y)(x.ptr, y.ptr)
        # # 3D
        # if x.ndim == 3:
        #     # (m, n, p) + (1, n, p)
        #     if y.shape[0] == 1 and x.shape[1:] == y.shape[1:]:
        #         arr = _add.lib.add_3d_with_2d(x.ptr, y.ptr, x.numel, y.numel)
        #     # (m, n, p) + (m, 1, p)
        #     elif x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2] and y.shape[1] == 1:
        #         arr = _add.lib.add_with_dim1_with_dim0(x.ptr, y.ptr, x.numel, y.numel, x.shape[1]*x.shape[2], x.shape[2])  # noqa: E501
        #     # (m, n, p) + (m, n, 1)
        #     elif x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] and y.shape[2] == 1:
        #         arr = _add.lib.add_with_dim0(x.ptr, y.ptr, x.numel, y.numel, x.shape[2])
        #     # (m, n, p) + (1, 1, p)
        #     elif y.shape[0] == 1 and y.shape[1] == 1 and x.shape[2] == y.shape[2]:
        #         arr = _add.lib.add_with_dim1(x.ptr, y.ptr, x.numel, x.shape[2])
        #     # (m, n, p) + (1, n, 1)
        #     elif y.shape[0] == 1 and y.shape[2] == 1 and x.shape[1] == y.shape[1]:
        #         arr = _add.lib.add_with_dim0(x.ptr, y.ptr, x.numel, y.numel, x.shape[2])
        #     # (m, n, p) + (m, 1, 1)
        #     elif x.shape[0] == y.shape[0] and y.shape[1] == 1 and y.shape[2] == 1:
        #         arr = _add.lib.add_with_dim0(x.ptr, y.ptr, x.numel, y.numel, x.shape[1]*x.shape[2])
        #     # (m, n, p) + (1, 1, 1)
        #     elif y.shape[0] == 1 and y.shape[1] == 1 and y.shape[2] == 1:
        #         arr = _add.lib.add_with_scalar(x.ptr, y.ptr, x.numel)
        #     # (m, n, p) + (m, n, p)
        #     elif x.shape == y.shape:
        #         arr = _add.lib.add(x.ptr, y.ptr, x.numel)
        # elif x.ndim == 2:
        #     # (m, n) + (1, n)
        #     if x.shape[1] == y.shape[1] and y.shape[0] == 1:
        #         arr = _add.lib.add_with_dim1(x.ptr, y.ptr, x.numel, x.shape[1])
        #     # (m, n) + (m, 1)
        #     elif x.shape[0] == y.shape[0] and y.shape[1] == 1:
        #         arr = _add.lib.add_with_dim0(x.ptr, y.ptr, x.numel, y.numel, x.shape[1])
        #     # (m, n) + (1, 1)
        #     elif y.shape[0] == 1 and y.shape[1] == 1:
        #         arr = _add.lib.add_with_scalar(x.ptr, y.ptr, x.numel)
        #     # (m, n) + (m, n)
        #     elif x.shape == y.shape:
        #         arr = _add.lib.add(x.ptr, y.ptr, x.numel)

        return CPU.ffi.gc(arr, _arithmetic.lib.free_op)

    @staticmethod
    @dispatcher.register(BinaryOps.MUL, Device.CPU)
    def mul(x: Buffer, y: Buffer) -> CDataPtr:
        pass

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.CPU)
    def sub(x: Buffer, y: Buffer) -> CDataPtr:
        if x.numel >= y.numel:
            main_buf, other_buf = x.ptr, y.ptr
            main_numel, other_numel = x.numel, y.numel
            main_stride, other_stride = x.stride, y.stride or [0] # for scalar
            main_shape, other_shape = x.shape, y.shape
            other_shape_len = len(y.shape)
        elif x.numel < y.numel:
            main_buf, other_buf = x.ptr, y.ptr
            main_numel, other_numel = x.numel, y.numel
            main_stride, other_stride = y.stride, x.stride or [0] # for scalar
            main_shape, other_shape = y.shape, x.shape
            other_shape_len = len(x.shape)

        if len(main_shape) == 2:
            arr = _arithmetic.lib.op_2d(main_buf, other_buf,
                                        main_numel, other_numel,
                                        main_shape, other_shape,
                                        main_stride, other_stride,
                                        other_shape_len, 1)
        elif len(main_shape) == 3:
            arr = _arithmetic.lib.op_3d(main_buf, other_buf,
                                        main_numel, other_numel,
                                        main_shape, other_shape,
                                        main_stride, other_stride,
                                        other_shape_len, 1)

        return CPU.ffi.gc(arr, _arithmetic.lib.free_op)

    @staticmethod
    @dispatcher.register(BinaryOps.NEG, Device.CPU)
    def neg(x: Buffer) -> CDataPtr:
        arr = _neg.lib.neg(x.ptr, x.numel)

        return CPU.ffi.gc(arr, _neg.lib.free_neg)

    @staticmethod
    @dispatcher.register(BinaryOps.MATMUL, Device.CPU)
    def matmul(x: Buffer, y: Buffer) -> CDataPtr:
        arr = _matmul.lib.matmul(x.ptr, y.ptr, x.shape[0], y.shape[1], y.shape[0], y.stride, x.stride)

        return CPU.ffi.gc(arr, _matmul.lib.free_matmul)

    @staticmethod
    @dispatcher.register(UnaryOps.SUM, Device.CPU)
    def sum(x: Buffer, dim: int | None, numel: int) -> CDataPtr:
        if x.ndim == 3:
            if dim == 0:
                arr = _sum.lib.sum_3d_dim0(x.ptr, numel, x.shape, x.stride)
            if dim == 1:
                arr = _sum.lib.sum_3d_dim1(x.ptr, numel, x.shape, x.stride)
            if dim == 2:
                arr = _sum.lib.sum_3d_dim2(x.ptr, numel, x.shape, x.stride)
            if not dim and dim != 0:
                arr = _sum.lib.sum(x.ptr, prod_(x.shape))
        if x.ndim == 2:
            if dim == 0:
                arr = _sum.lib.sum_2d_dim0(x.ptr, numel, x.shape, x.stride)
            if dim == 1:
                arr = _sum.lib.sum_2d_dim1(x.ptr, numel, x.shape, x.stride)
            if not dim and dim != 0:
                arr = _sum.lib.sum(x.ptr, prod_(x.shape))

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


"""
