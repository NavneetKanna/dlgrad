from __future__ import annotations

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dtype import DType, Scalar
from dlgrad.helpers import ffi, resolve_ndim
from dlgrad.runtime import (
	cpu,  # needed to register all the cpu runtime functions  # noqa: F401
)


class OP:
	"""
	This is the superclass for all the ops implemented in the ops module.
	Used by the autograd engine to build the graph.

	Thanks to tinygrad for the template, this is similar to the Function class.

	Attribute:
	    parents (tuple): A tuple containing the parents of the op.
	    requires_grad (bool): A bool to indicate whether the output Tensor should be used in backward pass.
	"""  # noqa: E501

	def __init__(self, *data: Tensor) -> None:
		self.parents: tuple = data
		self.req_grad = [i.requires_grad for i in data]
		self.requires_grad = True if any(self.req_grad) else False

	def forward(self, *args, **kwargs) -> None:
		raise NotImplementedError(f"forward not implemented for {type(self)}")  # noqa: ANN201

	def backward(self, *args, **kwargs) -> None:
		raise RuntimeError(f"backward not implemented for {type(self)}")  # noqa: ANN201

	def match_inp_shape(self, inp: Buffer, upstream_grad: Buffer) -> Buffer:
		inp_shape = inp.shape
		ndim = resolve_ndim(inp_shape=inp_shape, grad_shape=upstream_grad.shape)
		if not ndim:
			return upstream_grad

		for _ in range(ndim):
			upstream_grad = upstream_grad.sum(dim=0)

		if upstream_grad.shape != inp_shape:
			upstream_grad.metadata.shape = inp_shape
			upstream_grad.metadata.stride = inp.stride
			upstream_grad.metadata.numel = inp.numel

		return upstream_grad

	@classmethod
	def execute(cls: type[OP], *data: Tensor, **kwargs) -> Tensor:
		"""
		The main method that is called to execute an op.

		This method takes a subclass (cls) as parameter and calls its forward method.
		Then it returns a Tensor with the returned data and attributes.

		Parameters:
		    fxn (Type[OP]) : One of the Op's class defined in the ops module.
		    data (tuple(Tensor)) : A tuple of Tensors, which are the parents of the op.
		    **kwargs (dict) : Any additional keyword args.

		Returns:
		    Tensor: A Tensor which is the output of the op.
		"""
		ctx = cls(*data)
		tensor = Tensor.__new__(Tensor)
		tensor.data = ctx.forward(*[d.data for d in data], **kwargs)
		tensor.requires_grad = ctx.requires_grad
		tensor.dtype = kwargs.get("dtype", data[0].dtype)
		tensor.device = kwargs.get("device", data[0].device)
		tensor._ctx = ctx if ctx.requires_grad else None
		tensor.grad = None

		return tensor


import dlgrad.ops as ops  # since ops module imports OP class, it is placed after the defination  # noqa: E402, E501


class Tensor:
	def __init__(
		self,
		data: Scalar | Buffer | "np.ndarray",  # type: ignore  # noqa: F821
		device: str | Device | None = Device.CPU,  # noqa: F821 # type: ignore
		dtype: str | DType | None = None,
		requires_grad: bool = False,
	) -> None:
		self.device: Device = (
			device
			if isinstance(device, Device)
			else Device.from_str(device)
			if isinstance(device, str)
			else Device.CPU
		)
		self.dtype: DType = (
			dtype
			if isinstance(dtype, DType)
			else DType.from_str(dtype)
			if isinstance(dtype, str)
			else DType.FLOAT32
		)
		self.requires_grad: bool = requires_grad
		self._ctx: OP = None  # used by autograd engine
		self.grad = None

		if isinstance(data, Scalar):
			self.dtype = DType.get_dtype_from_py(data)
			self.data = Buffer.create_buffer_from_scalar(data, self.device)
		elif str(type(data)) == "<class 'numpy.ndarray'>":
			if str(data.dtype) != "float32":
				raise ValueError("dlgrad only supports float32 dtype")
			self.data = Buffer(
				ffi.from_buffer(
					cdecl="float *", python_buffer=data, require_writable=False
				),
				data.shape,
				device,
				ndim=data.ndim,
			)
		elif isinstance(data, Buffer):
			self.data = data

	def numpy(self: Tensor) -> "np.ndarray":  # type: ignore  # noqa: F821
		import numpy as np

		data = np.frombuffer(
			ffi.buffer(self.data.ptr, self.data.numel * ffi.sizeof("float")),
			count=-1,
			dtype=np.float32,
		)

		t = np.lib.stride_tricks.as_strided(
			data,
			self.data.shape,
			tuple(
				stride * DType.get_n_bytes(self.dtype) for stride in self.data.stride
			),
		)

		return t

	@staticmethod
	def uniform(
		shape: tuple,
		device: str | Device | None = Device.CPU,
		dtype: str | DType | None = DType.FLOAT32,
		low: float = 0.0,
		high: float = 1.0,
		**kwargs,
	) -> Tensor:
		"""
		Creates a Tensor with the specified shape filled with random numbers from a
		uniform distribution on the interval [low, high).

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
			raise NotImplementedError("dlgrad only supports float32")
		if not isinstance(shape, tuple):
			raise ValueError("shape must be a tuple")

		return Tensor(
			data=Buffer.uniform(shape, device=device, low=low, high=high),
			device=device,
			dtype=dtype,
			requires_grad=kwargs.get("requires_grad"),
		)

	@staticmethod
	def rand(
		shape: tuple,
		device: str | Device | None = Device.CPU,
		dtype: str | DType | None = DType.FLOAT32,
		**kwargs,
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
			raise NotImplementedError("dlgrad supports only float32")
		if not isinstance(shape, tuple):
			raise ValueError("shape must be a tuple")

		return Tensor.uniform(shape, device, dtype, **kwargs)

	@staticmethod
	def full(
		shape: tuple,
		fill_value: Scalar,
		device: Device = Device.CPU,
		dtype: DType = DType.FLOAT32,
		**kwargs,
	) -> Tensor:
		return Tensor(
			data=Buffer.full(shape, fill_value=fill_value, device=device),
			device=device,
			dtype=dtype,
			requires_grad=kwargs.get("requires_grad"),
		)

	@staticmethod
	def ones_like(
		shape: tuple,
		device: Device = Device.CPU,
		dtype: DType = DType.FLOAT32,
		**kwargs,
	) -> Tensor:
		return Tensor.full(shape, 1.0, device, dtype, **kwargs)

	@staticmethod
	def add(x: Tensor, y: Tensor) -> Tensor:
		return ops.Add.execute(x, y)

	@staticmethod
	def sub(x: Tensor, y: Tensor) -> Tensor:
		return ops.Sub.execute(x, y)

	@staticmethod
	def mul(x: Tensor, y: Tensor) -> Tensor:
		return ops.Mul.execute(x, y)

	@staticmethod
	def neg(x: Tensor) -> Tensor:
		return ops.Neg.execute(x)

	@staticmethod
	def matmul(x: Tensor, y: Tensor) -> Tensor:
		if (
			x.data.shape[-1] != y.data.shape[0]
			and x.data.ndim != 2
			and y.data.ndim != 2
		):
			raise ValueError("Either the Tensors shape dont match or is not 2D")

		return ops.MatMul.execute(x, y)

	@staticmethod
	def transpose(x: Tensor) -> Tensor:
		assert x.data.ndim == 2, "Only 2D Tensors can be transposed"

		return Tensor(
			data=ops.transpose(x.data),
			device=x.device,
			dtype=x.dtype,
			requires_grad=x.requires_grad,
		)

	def sum(self, dim: int | None = None) -> Tensor:
		return ops.Sum.execute(self, dim=dim)

	def relu(self) -> Tensor:
		ops.Relu.execute(self)

	def linear(self, weight: Tensor, bias: Tensor | None) -> Tensor:
		return self @ weight.T + bias if bias else self @ weight.T

	def backward(self) -> None:
		assert self.shape == tuple(), "backward must be called on a scalar Tensor"

		topo: list[Tensor] = []
		visited = set()

		# leaf tensors are not included
		def _topo_sort(node: Tensor):  # noqa: ANN202
			if node not in visited:
				visited.add(node)
				ctx = getattr(node, "_ctx", None)
				if ctx:
					for i in node._ctx.parents:
						_topo_sort(i)
					topo.append(node)

		_topo_sort(self)

		print("topo", topo)

		self.grad = Tensor(1.0)

		# TODO: del _ctx
		for node in reversed(topo):
			upstream_grads: tuple[Buffer] = node._ctx.backward(node.grad.data)
			upstream_grads: list[Tensor] = [
				Tensor(g, device=self.device, requires_grad=False)
				for g in upstream_grads
			]
			for p, g in zip(node._ctx.parents, upstream_grads):
				if p.requires_grad:
					assert (
						g.shape == p.shape
					), f"Tensor shape and grad shape must match {p.shape}, {g.shape}"
					p.grad = g if p.grad is None else p.grad + g

	def __repr__(self) -> str:
		return f"Tensor<dtype: {self.dtype} device: {self.device}, shape: {self.shape}, ndim: {self.ndim}>"  # noqa: E501

	@property
	def T(self) -> Tensor:  # noqa: N802
		return Tensor.transpose(self)

	def __add__(self, other: Tensor) -> Tensor:
		return Tensor.add(self, other)

	def __sub__(self, other: Tensor) -> Tensor:
		return Tensor.sub(self, other)

	def __neg__(self) -> Tensor:
		return Tensor.neg(self)

	def __matmul__(self, other: Tensor) -> Tensor:
		return Tensor.matmul(self, other)

	@property
	def numel(self) -> int:
		return self.data.numel

	@property
	def shape(self) -> tuple:
		return self.data.shape

	@property
	def stride(self) -> tuple:
		return self.data.stride

	@property
	def ndim(self) -> int:
		return self.data.ndim
