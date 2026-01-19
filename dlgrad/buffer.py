from __future__ import annotations

import sys
from dataclasses import dataclass

from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, DType, Scalar
from dlgrad.helpers import (
    BinaryOps,
    BufferOps,
    CustomOps,
    UnaryOps,
    broadcast_shapes,
    cal_sum_max_out_shape,
    calculate_stride,
    check_broadcast,
    ffi,
    get_broadcast_shape,
    prod_,
)
from dlgrad.runtime.cpu import CPU


@dataclass
class BufferMetadata:
    shape: tuple
    numel: int
    stride: tuple
    ndim: int
    dtype: DType
    device: Device
    nbytes: int


class Buffer:
    def __init__(
        self, data: CDataPtr, shape: tuple,
        device: str | Device | None = Device.CPU,
        dtype: str | DType | None = None, **kwargs
    ) -> None:
        self.ptr = data # ptr to the array
        self.metadata = BufferMetadata(shape=shape, numel=prod_(shape),
                                       stride=kwargs.get("stride", calculate_stride(shape)),
                                       ndim=kwargs.get("ndim", len(shape)),
                                       dtype=dtype, device=device, nbytes=prod_(shape)*DType.get_n_bytes(dtype))

    def tobytes(self) -> bytes:
        """
        Copies the raw C memory data into a Python bytes object.
        """
        return bytes(CPU.ffi.buffer(self.ptr, self.metadata.nbytes))

    def copy_from(self, source_bytes: bytes) -> None:
        """
        Copies raw Python bytes into the C memory pointer.
        """
        if len(source_bytes) != self.metadata.nbytes:
            raise ValueError(f"Buffer size mismatch! Expected {self.metadata.nbytes} bytes, got {len(source_bytes)}")

        CPU.ffi.memmove(self.ptr, source_bytes, len(source_bytes))

    def show(self: Buffer) -> None:
        dispatcher.dispatch(op=CustomOps.PRINT, device=Device.CPU, x=self)

    # def reshape(self, new_shape: tuple) -> None:
    #     self.metadata = BufferMetadata(shape=new_shape, numel=prod_(new_shape),
    #                                    stride=calculate_stride(new_shape), ndim=len(new_shape),
    #                                    dtype=self.dtype, device=self.device,
    #                                    nbytes=prod_(new_shape)*DType.get_n_bytes(self.dtype))

    def reshape(self, new_shape: tuple) -> "Buffer":
        if prod_(new_shape) != self.metadata.numel:
            raise ValueError(f"Cannot reshape {self.shape} ({self.metadata.numel} elements) to {new_shape} ({prod_(new_shape)} elements)")

        return Buffer(
            data=self.ptr,
            shape=new_shape,
            device=self.device,
            dtype=self.dtype
        )

    def numpy(self: Buffer) -> "np.ndarray":  # type: ignore  # noqa: F821
        import numpy as np

        data = np.frombuffer(
            ffi.buffer(self.ptr, self.numel * ffi.sizeof("float")),
            count=-1,
            dtype=np.float32,
        )

        t = np.lib.stride_tricks.as_strided(
            data,
            self.shape,
            tuple(
                stride * DType.get_n_bytes(self.dtype) for stride in self.stride
            ),
        )

        return t

    @staticmethod
    def from_scalar(val: Scalar) -> Buffer:
        float_arr = ffi.new("float[]", 1)
        float_arr[0] = val
        return Buffer(data=float_arr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)

    @staticmethod
    def uniform(shape: tuple, device: Device, dtype: DType | str, **kwargs) -> Buffer:
        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)
        if isinstance(device, str):
            device = Device.from_str(device)

        if dtype is not DType.FLOAT32:
            raise NotImplementedError("dlgrad only supports float32")
        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple")

        # All data creation is done on cpu
        return Buffer(
            data=dispatcher.dispatch(op=BufferOps.UNIFORM,
                                     device=Device.CPU if device != Device.CPU else device,
                                     shape=shape, **kwargs),
            shape=shape, device=device, dtype=dtype
        )

    @staticmethod
    def full(shape: tuple, fill_value: Scalar, device: Device, dtype: DType) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BufferOps.FULL, device=Device.CPU, shape=shape, fill_value=fill_value),
            shape=shape, device=device, dtype=dtype
        )

    @staticmethod
    def arange(shape: tuple, device: Device, dtype: DType = DType.FLOAT32) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BufferOps.ARANGE, device=Device.CPU, shape=shape),
            shape=shape, device=device, dtype=dtype
        )

    @staticmethod
    def masked_fill(x: Buffer, mask: Buffer, val: Scalar) -> Buffer:
        out_shape = get_broadcast_shape(x.shape, mask.shape)
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.MASKED_FILL, device=Device.CPU, x=x, mask=mask, val=val, out_shape=out_shape),
            shape=out_shape, device=x.device, dtype=x.dtype
        )

    def unsqueeze(self, dim: list[int] | int) -> None:
        if dim == -1:
            dim = 0
        if isinstance(dim, int):
            dim = [dim]
        if len(set(dim)) != len(dim):
            raise AssertionError(f"duplicate dims not allowed in unsqueeze: {dim}")
        for i in dim:
            assert i <= len(self.shape), f"Cannot unsqueeze {dim} from {self.shape}"

        new_shape = list(self.shape)
        for d in sorted(dim, reverse=True):
            new_shape.insert(d, 1)

        self.metadata.shape = tuple(new_shape)
        self.metadata.numel = prod_(new_shape)
        self.metadata.stride = calculate_stride(new_shape)
        self.metadata.ndim = len(new_shape)

    def squeeze(self, dim: list[int] | int) -> None:
        if isinstance(dim, int):
            dim = [dim]
        for idx in dim:
            if self.shape[idx] != 1:
                continue

            new_shape = list(self.shape)
            new_shape.pop(idx)
            self.metadata.shape = tuple(new_shape)
            self.metadata.numel = prod_(new_shape)
            self.metadata.stride = calculate_stride(new_shape)
            self.metadata.ndim = len(new_shape)

    def sum(self, dim: int = -1, keepdim: bool = False) -> Buffer:
        out_shape = cal_sum_max_out_shape(ndim=self.ndim, dim=dim, inp_shape=self.shape, keepdim=keepdim)

        if keepdim:
            ndim = self.ndim
        else:
            if dim == -1:
                ndim = 0
            else:
                ndim = self.ndim - 1

        # print(self.shape, dim, self.ndim)
        if self.ndim == 1:
            d = Device.CPU
        elif dim in range(0, self.shape[-1]+1):
            d = Device.CPU
        else:
            d = self.device

        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.SUM, device=d, x=self, dim=dim),
            shape=out_shape, device=self.device,
            ndim=ndim, dtype=self.dtype
        )

    def mean(self, dim: int = -1, keepdim: bool = False) -> Buffer:
        out_shape = cal_sum_max_out_shape(ndim=self.ndim, dim=dim, inp_shape=self.shape, keepdim=keepdim)

        if keepdim:
            ndim = self.ndim
        else:
            if dim == -1:
                ndim = 0
            else:
                ndim = self.ndim - 1

        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.MEAN, device=Device.CPU, x=self, dim=dim),
            shape=out_shape, device=self.device,
            ndim=ndim, dtype=self.dtype
        )

    def max(self, dim: int = -1, out: Buffer = None, keepdim: bool = False) -> Buffer:
        out_shape = cal_sum_max_out_shape(ndim=self.ndim, dim=dim, inp_shape=self.shape, keepdim=keepdim)

        if dim in range(0, self.shape[-1]+1):
            d = Device.CPU
        else:
            d = self.device
        out = dispatcher.dispatch(op=UnaryOps.MAX, device=d, x=self, dim=dim)

        if keepdim:
            ndim = self.ndim
        else:
            if dim == -1:
                ndim = 0
            else:
                ndim = self.ndim - 1

        out_buf = Buffer(data=out, shape=out_shape, device=self.device, ndim=ndim, dtype=self.dtype)  # noqa: E501

        return out_buf

    def matmul(self, other: Buffer) -> Buffer:
        p1, p2, out_shape = broadcast_shapes(self.shape, other.shape)

        # Create temporary reshaped buffers if needed (sharing data pointers)
        x = Buffer(data=self.ptr, shape=p1, device=self.device, dtype=self.dtype) if self.shape != p1 else self
        y = Buffer(data=other.ptr, shape=p2, device=other.device, dtype=other.dtype) if other.shape != p2 else other

        M, K, N = p1[-2], p1[-1], p2[-1]
        device = self.device
        if sys.platform == "darwin" and all(dim % 8 == 0 for dim in (M, K, N)):
            device = Device.METAL

        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.MATMUL, device=device, x=x, y=y),
            shape=out_shape,
            device=self.device,
            dtype=self.dtype
        )

    def _matmul(self, other: Buffer) -> Buffer:
        p1, p2, out_shape = broadcast_shapes(self.shape, other.shape)

        old_self_shape = self.shape
        old_other_shape = other.shape

        if self.shape != p1:
            self.reshape(p1)

        if other.shape != p2:
            other.reshape(p2)

        M, K, N = p1[-2], p1[-1], p2[-1]

        device = self.device
        if sys.platform == "darwin" and all(dim % 8 == 0 for dim in (M, K, N)):
            device = Device.METAL

        t = Buffer(
            data=dispatcher.dispatch(op=BinaryOps.MATMUL, device=device, x=self, y=other),
            shape=out_shape,
            device=self.device,
            dtype=self.dtype
        )

        self.reshape(old_self_shape)
        other.reshape(old_other_shape)

        return t

        # assert self.ndim == other.ndim, f"self shape ({self.shape}) does not match other shape ({other.shape})"
        # assert self.ndim == 2 and other.ndim == 2, "dlgrad only supports 2d matrix multiplication"
        # if (self.shape[-1] != other.shape[0] and self.ndim != 2 and other.ndim != 2):
        # raise ValueError("Either the Tensors shape dont match or is not 2D")
        if self.ndim == 4:
            device = self.device
            B = max(self.shape[0], other.shape[0])
            C = max(self.shape[1], other.shape[1])
            M = self.shape[2]
            N = other.shape[3] # Corrected N

            shape = (B, C, M, N)
            # if self.shape[0] == 1:
            #     shape = (other.shape[0], self.shape[1], self.shape[2], other.shape[3])
            # else:
            #     shape = (self.shape[0], self.shape[1], self.shape[2], other.shape[3])
        elif self.ndim == 3:
            device = self.device
            if self.shape[0] == 1:
                shape = (other.shape[0], self.shape[1], other.shape[2])
            else:
                # shape = (self.shape[0], self.shape[1], other.shape[2])
                shape = (self.shape[0], self.shape[1], other.shape[-1])
        elif self.shape[0] % 8 == 0 and self.shape[1] % 8 == 0 and other.shape[0] % 8 == 0 and other.shape[1] % 8 == 0 and sys.platform == "darwin":
            device = Device.METAL
            shape = (self.shape[0], other.shape[1])
        else:
            device = Device.CPU
            shape = (self.shape[0], other.shape[1])

        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.MATMUL, device=device, x=self, y=other),
            shape=shape, device=self.device, dtype=self.dtype
        )

    @staticmethod
    def swap_indices(inp: tuple, tmp: tuple) -> tuple:
        lst = list(inp)
        for i in range(0, len(tmp), 2):
            if i+1 < len(tmp):
                idx1, idx2 = tmp[i], tmp[i+1]
                lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
            return tuple(lst)

    def transpose(self, dim0: int, dim1: int) -> Buffer:
        # assert self.ndim == 2, "Only 2D Tensors can be transposed"

        if self.ndim == 4 and ((dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1)):
            out_shape = (self.shape[0], self.shape[2], self.shape[1], self.shape[3])
        elif self.ndim == 4 and ((dim0 == 2 and dim1 == 3) or (dim0 == 3 and dim1 == 2)):
            out_shape = (self.shape[0], self.shape[1], self.shape[3], self.shape[2])
        elif self.ndim == 3 and ((dim0 == 0 and dim1 == 1) or (dim0 == 1 and dim1 == 0)):
            out_shape = (self.shape[1], self.shape[0], self.shape[2])
        elif self.ndim == 3 and ((dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1)):
            out_shape = (self.shape[0], self.shape[2], self.shape[1])
        else:
            out_shape = (self.shape[1], self.shape[0])

        return Buffer(
            data=dispatcher.dispatch(
                op=UnaryOps.TRANSPOSE,
                device=self.device,
                x=self,
                dim0=dim0,
                dim1=dim1,
                out_stride=calculate_stride(out_shape),
                out_shape=out_shape
            ),
            shape=out_shape, device=self.device, dtype=self.dtype
        )

    def exp(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.EXP, device=self.device if self.ndim !=0 else Device.CPU, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def sqrt(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.SQRT, device=self.device if self.ndim !=0 else Device.CPU, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def rsqrt(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.RSQRT, device=Device.CPU, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def clamp(self, min: int | None = None, max: int | None = None) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.CLAMP, device=Device.CPU, x=self, min=min, max=max),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def log(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.LOG, device=self.device if self.ndim !=0 else Device.CPU, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def relu(self) -> Buffer:
        return Buffer(
            data=self.where(inp=self, other=0.0).ptr,
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def sigmoid(self) -> Buffer:
        out = self.exp() / (self.exp() + 1.0)
        return Buffer(
            data=out.ptr,
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def leaky_relu(self, neg_slope: Scalar = 0.01) -> Buffer:
        return Buffer(
            data=self.where(inp=self, other=self*neg_slope).ptr,
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def tanh(self) -> Buffer:
        out = (self.exp() - (-self).exp()) / (self.exp() + (-self).exp())
        return Buffer(
            data=out.ptr,
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    @staticmethod
    def ce_forward(x: Buffer, y: Buffer) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=CustomOps.CE_FORWARD, device=Device.CPU,
                                     x=x, y=y),
            shape=(1, x.shape[0]), device=x.device, dtype=x.dtype
        )

    def argmax(self, dim: int) -> Buffer:
        assert self.ndim == 2, "currently dlgrad supports argmax for 2d only"
        if dim==0:
            s = (1, self.shape[1])
        elif dim==1:
            s = (self.shape[0], 1)
        else:
            s = (1, 1)
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.ARGMAX, device=Device.CPU,
                                     x=self, dim=dim),
            shape=s, device=self.device, dtype=self.dtype
        )

    def where(self, inp: Buffer | Scalar, other: Buffer | Scalar) -> Buffer:
        if isinstance(inp, Scalar):
            inp = Buffer.from_scalar(inp)
        if isinstance(other, Scalar):
            other = Buffer.from_scalar(other)
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.WHERE, device=self.device,
                                     x=self, inp=inp, other=other),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    @staticmethod
    def ce_backward(**kwargs) -> None:
        kwargs.pop("device")
        dispatcher.dispatch(op=kwargs.pop("op"), device=Device.CPU, **kwargs)

    def _binary_op(self, other: Buffer, op: BinaryOps) -> Buffer:
        if not check_broadcast(self.shape, other.shape):
            raise ValueError(f"Cannot broadcast {other.shape} to {self.shape}")

        output_shape = self.shape if self.numel >= other.numel else other.shape

        if op == BinaryOps.SUB and self.numel < other.numel:
            x, y = (self, other) if self.numel >= other.numel else (other, self)
            # if y.shape is different from x.shape, for example, (2, 3, 4) & (3, 4)
            # unsqueeze y.shape to (1, 3, 4) and squeeze it back to (3, 4)
            tmp = []
            org_yshape = y.shape
            if len(x.shape) != len(y.shape) and y.ndim != 1 and y.ndim != 0:
                shape_diff = len(x.shape) - len(y.shape)
                for i in range(shape_diff):
                    tmp.append(i)
                    y.unsqueeze(0)
            tmp = Buffer(
                data=dispatcher.dispatch(op=BinaryOps.SUB, device=self.device, x=other, y=self),
                shape=other.shape, device=self.device, dtype=self.dtype
            )
            return Buffer(
                data=dispatcher.dispatch(op=UnaryOps.NEG, device=Device.CPU, x=tmp),
                shape=tmp.shape, device=tmp.device, dtype=self.dtype
            )

        x, y = (self, other) if self.numel >= other.numel else (other, self)

        # if y.shape is different from x.shape, for example, (2, 3, 4) & (3, 4)
        # unsqueeze y.shape to (1, 3, 4) and squeeze it back to (3, 4)
        tmp = []
        org_yshape = y.shape
        if len(x.shape) != len(y.shape) and y.ndim != 1 and y.ndim != 0:
            shape_diff = len(x.shape) - len(y.shape)
            for i in range(shape_diff):
                tmp.append(i)
                y.unsqueeze(0)

        t = Buffer(
            data=dispatcher.dispatch(op=op, device=self.device, x=x, y=y),
            shape=output_shape, device=self.device, dtype=self.dtype
        )

        if len(x.shape) != len(org_yshape) and y.ndim != 1 and y.ndim != 0:
            y.squeeze(tmp)

        return t

    def embedding(self, idx: Buffer, backward: bool = False, upstream_grad: Buffer = None) -> Buffer:
        if backward:
            out_shape = self.shape
            return Buffer(
                data=dispatcher.dispatch(op=CustomOps.EMBEDDING, device=Device.CPU, x=self, idx=idx, out_numel=prod_(out_shape), backward=True, upstream_grad=upstream_grad),
                shape=out_shape, device=self.device, dtype=self.dtype
            )
        out_shape = idx.shape + (self.shape[-1],)
        return Buffer(
            data=dispatcher.dispatch(op=CustomOps.EMBEDDING, device=Device.CPU, x=self, idx=idx, out_numel=prod_(out_shape)),
            shape=out_shape, device=self.device, dtype=self.dtype
        )


    def __add__(self, other: Buffer | Scalar) -> Buffer:
        if isinstance(other, Scalar):
            other = Buffer(data=Buffer.from_scalar(other).ptr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)
        return self._binary_op(other, BinaryOps.ADD)

    def __sub__(self, other: Buffer | Scalar) -> Buffer:
        if isinstance(other, Scalar):
            other = Buffer(data=Buffer.from_scalar(other).ptr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)
        return self._binary_op(other, BinaryOps.SUB)

    def __mul__(self, other: Buffer | Scalar) -> Buffer:
        if isinstance(other, Scalar):
            other = Buffer(data=Buffer.from_scalar(other).ptr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)
        return self._binary_op(other, BinaryOps.MUL)

    def __truediv__(self, other: Buffer | Scalar) -> Buffer:
        if isinstance(other, Scalar):
            other = Buffer(data=Buffer.from_scalar(other).ptr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)

        if self.numel >= other.numel:
            return self._binary_op(other, BinaryOps.DIV)
        else:
            return self._binary_op(other**-1, BinaryOps.MUL)

    def __rsub__(self: Buffer, other: Scalar) -> Buffer:
        if isinstance(other, Scalar):
            other = Buffer(data=Buffer.from_scalar(other).ptr, shape=(), device=Device.CPU, dtype=DType.FLOAT32)

        return -(self._binary_op(other, BinaryOps.SUB))

    def __pow__(self, val: Scalar) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.POW, device=self.device if self.ndim !=0 else Device.CPU, x=self, val=val),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __le__(self, other: Buffer) -> Buffer:
        # TODO: Check broadcast, if self.numel < other.numel
        out_shape = get_broadcast_shape(self.shape, other.shape)
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.CMP, device=Device.CPU, x=self, y=other,out_shape=out_shape, mode="<="),
            shape=out_shape, device=self.device, dtype=self.dtype
        )

    def __gt__(self, other: Scalar) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.GT, device=Device.CPU, x=self, y=other),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __ge__(self, other: Scalar) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.GTE, device=Device.CPU, x=self, y=other),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __neg__(self) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=UnaryOps.NEG, device=self.device if self.ndim !=0 else Device.CPU, x=self),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __eq__(self, other: Buffer) -> Buffer:
        return Buffer(
            data=dispatcher.dispatch(op=BinaryOps.EQT, device=Device.CPU, x=self, y=other),
            shape=self.shape, device=self.device, dtype=self.dtype
        )

    def __matmul__(self, other: Buffer) -> Buffer:
        return self.matmul(other)

    @property
    def T(self) -> Buffer:  # noqa: N802
        return self.transpose(0, 1)

    @property
    def numel(self) -> int:
        return self.metadata.numel

    @property
    def shape(self) -> tuple:
        return self.metadata.shape

    @property
    def stride(self) -> tuple:
        return self.metadata.stride

    @property
    def ndim(self) -> int:
        return self.metadata.ndim

    @property
    def dtype(self) -> int:
        return self.metadata.dtype

    @property
    def device(self) -> int:
        return self.metadata.device

    @property
    def nbytes(self) -> int:
        return self.metadata.nbytes
