from __future__ import annotations

from functools import reduce

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

	def match_inp_shape(self, inp: Buffer, upstream_grad: Buffer, dim: int = 0) -> Buffer:
		inp_shape = inp.shape
		ndim = resolve_ndim(inp_shape=inp_shape, grad_shape=upstream_grad.shape)
		if not ndim:
			return upstream_grad

		for _ in range(ndim):
			upstream_grad = upstream_grad.sum(dim=dim)

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
		tensor.data.metadata.dtype = kwargs.get("dtype", data[0].dtype)
		tensor.data.metadata.device = kwargs.get("device", data[0].device)
		tensor.requires_grad = ctx.requires_grad
		tensor._ctx = ctx if ctx.requires_grad else None
		tensor.grad = None

		return tensor


import dlgrad.ops as ops  # since ops module imports OP class, it is placed after the defination  # noqa: E402, E501


class Tensor:
	def __init__(self, data: Buffer | "np.ndarray" | Scalar,  # type: ignore  # noqa: F821
			 	 requires_grad: bool = False) -> None:
		self.requires_grad: bool = requires_grad
		self._ctx: OP = None  # used by autograd engine
		self.grad = None

		if str(type(data)) == "<class 'numpy.ndarray'>":
			if str(data.dtype) != "float32":
				raise ValueError("dlgrad only supports float32 dtype")

			shape = data.shape
			ndim = data.ndim
			if len(data.shape) == 1:
				shape = (1,) + data.shape
			elif not shape:
				shape = (1, 1)
				ndim = 2

			self.data = Buffer(
				data=ffi.from_buffer(cdecl="float *", python_buffer=data, require_writable=False),
				shape=shape, dtype=DType.FLOAT32, device=Device.CPU, ndim=ndim
			)
		elif isinstance(data, Buffer):
			self.data = data
		elif isinstance(data, Scalar):
			self.data = Buffer.from_scalar(data)
		else:
			raise ValueError("The data must be of type Buffer, np.ndarray or float")

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
	def uniform(shape: tuple, device: str | Device | None = Device.CPU,
				dtype: str | DType | None = DType.FLOAT32, low: float = 0.0,
				high: float = 1.0, **kwargs) -> Tensor:
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
		return Tensor(
			data=Buffer.uniform(shape, device=device, dtype=dtype, low=low, high=high),
			requires_grad=kwargs.get("requires_grad"),
		)

	@staticmethod
	def rand(shape: tuple, device: str | Device | None = Device.CPU,
		     dtype: str | DType | None = DType.FLOAT32, **kwargs) -> Tensor:
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
		return Tensor.uniform(shape, device, dtype, **kwargs)

	@staticmethod
	def full(shape: tuple, fill_value: Scalar, device: Device = Device.CPU,
			 dtype: DType = DType.FLOAT32, **kwargs) -> Tensor:
		return Tensor(
			data=Buffer.full(shape, fill_value=fill_value, device=device, dtype=dtype),
			requires_grad=kwargs.get("requires_grad")
		)

	@staticmethod
	def ones_like(shape: tuple, device: Device = Device.CPU,
			      dtype: DType = DType.FLOAT32, **kwargs) -> Tensor:
		return Tensor.full(shape, 1.0, device, dtype, **kwargs)

	@staticmethod
	def zeros_like(shape: tuple, device: Device = Device.CPU,
			      dtype: DType = DType.FLOAT32, **kwargs) -> Tensor:
		return Tensor.full(shape, 0.0, device, dtype, **kwargs)

	@staticmethod
	def add(x: Tensor, y: Tensor | Scalar) -> Tensor:
		if isinstance(y, Scalar):
			y = Tensor(y)
		return ops.Add.execute(x, y)

	@staticmethod
	def mul(x: Tensor, y: Tensor | Scalar) -> Tensor:
		if isinstance(y, Scalar):
			y = Tensor(y)
		return ops.Mul.execute(x, y)

	@staticmethod
	def sub(x: Tensor, y: Tensor | Scalar) -> Tensor:
		if isinstance(y, Scalar):
			y = Tensor(y)
		return ops.Sub.execute(x, y)

	@staticmethod
	def div(x: Tensor, y: Tensor | Scalar) -> Tensor:
		if isinstance(y, Scalar):
			y = Tensor(y)
		return ops.Div.execute(x, y)

	@staticmethod
	def matmul(x: Tensor, y: Tensor) -> Tensor:
		return ops.MatMul.execute(x, y)

	@staticmethod
	def transpose(x: Tensor) -> Tensor:
		return ops.Transpose.execute(x)

	def sum(self, dim: int = -1) -> Tensor:
		return ops.Sum.execute(self, dim=dim)

	def relu(self) -> Tensor:
		return ops.Relu.execute(self)

	def linear(self, weight: Tensor, bias: Tensor | None) -> Tensor:
		return self @ weight.T + bias if bias else self @ weight.T

	def max(self, dim: int = -1) -> Tensor:
		return ops.Max.execute(self, dim=dim)

	def exp(self) -> Tensor:
		return ops.Exp.execute(self)

	def log(self) -> Tensor:
		return ops.Log.execute(self)

	def sqrt(self) -> Tensor:
		return ops.Sqrt.execute(self)

	def log_softmax(self, dim: int = 1) -> Tensor:
		t = self.max(dim=dim)
		m = self - t
		e = m.exp()
		ss = e.sum(dim=dim)
		return m - ss.log()

	def sequential(self, layers: list[callable[Tensor]]) -> None:
		return reduce(lambda inp, layer: layer(inp), layers, self)

	def cross_entropy_loss(self, target: Tensor) -> Tensor:
		if isinstance(target, Scalar):
			target = Tensor(target)
		return ops.CrossEntropy.execute(self, target)

	def backward(self) -> None:
		assert self.shape == (1, 1), "backward must be called on a scalar Tensor"

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

		self.grad = Tensor(1.0)

		# TODO: del _ctx
		for node in reversed(topo):
			if not node.grad:
				raise RuntimeError(f"Tensor {node} has no grad")
			upstream_grads: tuple[Buffer] = node._ctx.backward(node.grad.data)
			upstream_grads: list[Tensor] = [
				Tensor(g, requires_grad=False) for g in upstream_grads
			]
			for p, g in zip(node._ctx.parents, upstream_grads):
				p.grad = g if not p.grad else p.grad + g

	def __getitem__(self, idx: slice):  # noqa: ANN001, ANN204
		if isinstance(idx, slice) and idx.start is not None and idx.stop is not None and idx.step is None:
			s = idx.start*self.stride[0]
			ns = tuple([idx.stop-idx.start, *self.shape[1:]])

			buf = Buffer(data=self.data.ptr+s, shape=ns, device=self.device, dtype=self.dtype)

			return Tensor(data=buf, requires_grad=self.requires_grad)

	def __repr__(self) -> str:
		return f"Tensor<dtype: {self.dtype} device: {self.device}, shape: {self.shape}, ndim: {self.ndim}>"  # noqa: E501

	@property
	def T(self) -> Tensor:  # noqa: N802
		return Tensor.transpose(self)

	def __gt__(self, other: int | float) -> Tensor:
		return Tensor(data=self.data>other)

	def __add__(self, other: Tensor| Scalar) -> Tensor:
		return Tensor.add(self, other)

	# TODO: Support rmul, lmul
	def __mul__(self, other: Tensor | Scalar) -> Tensor:
		return Tensor.mul(self, other)

	def __sub__(self, other: Tensor | Scalar) -> Tensor:
		return Tensor.sub(self, other)

	def __truediv__(self, other: Tensor | Scalar) -> Tensor:
		return Tensor.div(self, other)

	def __pow__(self, val: int) -> Tensor:
		return Tensor(data=self.data**val)

	def __neg__(self) -> Tensor:
		return Tensor(data=-self.data, requires_grad=self.requires_grad)

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

	@property
	def dtype(self) -> int:
		return self.data.dtype

	@property
	def device(self) -> int:
		return self.data.device

"""
Roadmap:

- memory pool instead of malloc if malloc is slow to call repeatedly

"""
