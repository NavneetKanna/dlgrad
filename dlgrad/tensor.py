from __future__ import annotations

from dataclasses import dataclass
from typing import Type, get_args

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dtype import DType, Scalar
from dlgrad.helpers import calculate_stride, ffi, prod_
from dlgrad.runtime import \
    cpu  # needed to register all the cpu runtime functions  # noqa: F401


class OP:
    """
    This is the superclass for all the ops implemented in the ops module. 
    Used by the autograd engine to build the graph.

    Thanks to tinygrad for the template, this is similar to the Function class.

    Attribute:
        parents (tuple) : A tuple containing the parents of the op.
        requires_grad (bool) : A bool to indicate whether the output Tensor should be used in backward pass.
    """
    def __init__(self, *data: Tensor) -> None:
        self.parents: tuple = data
        req_grad = [i.requires_grad for i in data]
        self.requires_grad = True if any(req_grad) else False
    
    def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

    @staticmethod
    def _get_metadata(data: tuple[Tensor]) -> TensorMetadata:
        """
        Helper method to determine the metadata for the resulting tensor.

        Parameters:
            data (Tuple[Tensor]): A tuple of input Tensors.

        Returns:
            TensorMetadata: Metadata for the resulting tensor.
        """
        if len(data) > 1:
            tensor = data[0] if data[0].ndim >= data[1].ndim else data[1]
        else:
            tensor = data[0]
        
        return TensorMetadata(
            shape=tensor.shape,
            numel=tensor.numel,
            stride=tensor.stride,
            ndim=tensor.ndim
        )

    @classmethod
    def execute(fxn: Type[OP], *data: Tensor, **kwargs) -> Tensor:
        """
        The main method that is called to execute an op. 

        This method takes a subclass (cls) as parameter and calls its forward method. 
        Then it returns a Tensor with the returned data and attributes.

        Parameters:
            fix (Type[OP]) : One of the Op's class defined in the ops module.
            data (tuple(Tensor)) : A tuple of Tensors, which are the parents of the op.
            **kwargs (dict) : Any additional keyword args.

        Returns:
            Tensor: A Tensor which is the output of the op.
        """
        ctx = fxn(*data)
        tensor = Tensor.__new__(Tensor)
        tensor.data = ctx.forward(*data) 
        tensor.requires_grad  = ctx.requires_grad
        tensor.dtype = kwargs.get("dtype", data[0].dtype)
        tensor.device = kwargs.get("device", data[0].device)
        tensor._ctx = ctx if ctx.requires_grad else None 
        tensor.metadata = OP._get_metadata(data)

        return tensor


import dlgrad.ops as Op  # since ops module imports OP class, it is placed after the defination  # noqa: E402


@dataclass
class TensorMetadata:
    shape: tuple
    numel: int
    stride: tuple
    ndim: int


class Tensor:
    def __init__(
        self, data: Scalar | Buffer | 'np.ndarray', device: str | Device | None = None,  # noqa: F821 # type: ignore
        dtype: str | DType | None = None, requires_grad: bool = False, metadata: TensorMetadata = None
    ) -> None:
        self.device: Device = device if isinstance(device, Device) else Device.from_str(device) if isinstance(device, str) else Device.CPU
        self.dtype: DType = dtype if isinstance(dtype, DType) else DType.from_str(dtype) if isinstance(dtype, str) else DType.FLOAT32
        self.requires_grad: bool = requires_grad
        self._ctx: OP = None # used by autograd engine
        self.grad = None
        self.metadata = metadata

        if isinstance(data, get_args(Scalar)):
            self.dtype = DType.get_dtype_from_py(data)
            self.data = Op.create_buffer_from_scalar(data, dtype=self.dtype, device=self.device)
        elif str(type(data)) == "<class 'numpy.ndarray'>":
            if str(data.dtype) != "float32":
                raise ValueError("dlgrad only supports float32 dtype")

            self.data = Buffer(ffi.from_buffer(cdecl="float *", python_buffer=data, require_writable=False))
            self.metadata = TensorMetadata(data.shape, prod_(data.shape), calculate_stride(data.shape), data.ndim)
        elif isinstance(data, Buffer):
            self.data = data

    @staticmethod
    def rand(
        shape: tuple, device: str | Device | None = Device.CPU, 
        dtype: str | DType | None = DType.FLOAT32, **kwargs
    ) -> Tensor:
        """
        Creates a Tensor with the specified shape filled with random numbers from a 
        uniform distribution on the interval [0, 1).

        Parameters:
            shape (tuple) : The desired shape
            device (str | Device | None) : Default device is CPU
            dtype (str | DType | None) : Default dtype is float32
            **kwargs (dict) : Any additional keyword args.
        
        Returns:
            Tensor: A Tensor filled with random numbers.
        """
        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)

        if dtype is not DType.FLOAT32:
            raise NotImplementedError("rand is implemented only for float32")

        return Tensor(
            data=Op.uniform(shape, device=device), 
            device=device, 
            dtype=dtype, 
            requires_grad=kwargs.get("requires_grad"),
            metadata=TensorMetadata(shape, prod_(shape), calculate_stride(shape), len(shape))
        )
    
    def numpy(self: Tensor):
        import numpy as np

        return np.frombuffer(ffi.buffer(self.data.ptr, self.numel*ffi.sizeof("float")), count=-1, dtype=np.float32).reshape(self.shape)
    
    @staticmethod
    def add(x: Tensor, y: Tensor) -> Tensor:
        return Op.Add.execute(x, y)

    @staticmethod
    def sub(x: Tensor, y: Tensor) -> Tensor:
        return Op.Add.execute(x, -y)

    @staticmethod
    def neg(x: Tensor) -> Tensor:
        return Op.Neg.execute(x)

    @staticmethod
    def matmul(x: Tensor, y: Tensor) -> Tensor:
        return Op.MatMul.execute(x, y)

    def __repr__(self) -> str:
        return f"Tensor<dtype: {self.dtype} device: {self.device}>"

    @property
    def numel(self):
        return self.metadata.numel

    @property
    def shape(self):
        return self.metadata.shape
    
    @property
    def stride(self):
        return self.metadata.stride
    
    @property
    def ndim(self):
        return self.metadata.ndim

    def __add__(self, other):
        return Tensor.add(self, other)

    def __sub__(self, other):
        return Tensor.sub(self, other)

    def __neg__(self):
        return Tensor.neg(self)

    def __matmul__(self, other):
        return Tensor.matmul(self, other)
