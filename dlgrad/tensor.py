from __future__ import annotations

import platform
from functools import reduce

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dtype import DType, Scalar
from dlgrad.helpers import ffi
from dlgrad.runtime import cpu  # needed to register all the cpu runtime functions  # noqa: F401

if platform.system() == 'Darwin':
    from dlgrad.runtime import metal  # needed to register all the metal runtime functions # noqa: F401

class OP:
	"""
	This is the superclass for all the ops implemented in the ops module.
	Used by the autograd engine to build the graph.

	Thanks to tinygrad for the template.

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

	def reduce_grad_for_broadcasting(self, grad: Buffer, target_shape: tuple) -> Buffer:
		"""Reduce gradient to match target shape by summing over broadcasted dimensions"""

		current_shape = grad.shape

		# Handle different number of dimensions
		ndim_diff = len(current_shape) - len(target_shape)

		# Sum over extra dimensions
		for _ in range(ndim_diff):
			grad = grad.sum(dim=0)

		for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
			if target_dim == 1 and grad_dim > 1:
				grad = grad.sum(dim=i, keepdim=True)

		return grad

	@classmethod
	def execute(cls: type[OP], *data: Tensor, **kwargs) -> Tensor:
		"""
		The main method that is called to execute an op.

		This method takes a subclass (cls) as parameter and calls its forward method.
		Then it returns a Tensor with the returned data and attributes.

		Parameters
		----------
		fxn : Type[OP]
			One of the Op's class defined in the ops module.
		data : tuple(Tensor)
			A tuple of Tensors, which are the parents of the op.
		**kwargs : dict
			Any additional keyword args.

		Returns:
		    A Tensor which is the output of the op.
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
	# __slots__ = ("_ctx", "grad", "requires_grad", "data") # Not seeing any difference in speed

	def __init__(
			self, data: Buffer | "np.ndarray" | Scalar,  # type: ignore  # noqa: F821
			requires_grad: bool = False, device: Device | str = Device.CPU
		) -> None:
		self.requires_grad: bool = requires_grad
		self._ctx: OP = None  # used by autograd engine
		self.grad = None

		if str(type(data)) == "<class 'numpy.ndarray'>":
			if str(data.dtype) != "float32":
				raise ValueError("dlgrad only supports float32 dtype")
			if str(data.ndim) > '4':
				raise ValueError("dlgrad only supports upto 4D tensors")

			shape = data.shape
			ndim = data.ndim

			self.data = Buffer(
				data=ffi.from_buffer(cdecl="float *", python_buffer=data, require_writable=False),
				shape=shape, dtype=DType.FLOAT32, device=Device.from_str(device) if isinstance(device, str) else device, ndim=ndim
			)
		elif isinstance(data, Buffer):
			self.data = data
			self.data.metadata.device = Device.from_str(device) if isinstance(device, str) else device
		elif isinstance(data, Scalar):
			self.data = Buffer.from_scalar(data)
			self.data.metadata.device = Device.from_str(device) if isinstance(device, str) else device
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
				high: float = 1.0, requires_grad: bool = False) -> Tensor:
		"""
		Creates a tensor with the specified shape filled with random numbers from a
		uniform distribution on the interval [low, high).

		Parameters
		----------
		shape : tuple
				The desired shape
		device : str | Device | None
				Default device is CPU
		dtype : str | DType | None
				Default dtype is float32
		low : float
				The minimum value
		high : float
				The maximum value
		requires_grad : bool
				Default is False

		Returns:
		    A tensor filled with random numbers.
		"""
		return Tensor(
			data=Buffer.uniform(shape, device=device, dtype=dtype, low=low, high=high),
			requires_grad=requires_grad, device=device
		)

	@staticmethod
	def rand(shape: tuple, device: str | Device | None = Device.CPU,
		     dtype: str | DType | None = DType.FLOAT32, requires_grad: bool = False) -> Tensor:
		"""
		Creates a tensor with the specified shape filled with random numbers from a
		uniform distribution on the interval [0, 1).

		Parameters
		----------
		shape : tuple
				The desired shape
		device : str | Device | None
				Default device is CPU
		dtype : str | DType | None
				Default dtype is float32
		requires_grad : bool
				Default is False

		Returns:
		    A tensor filled with random numbers.
		"""
		return Tensor.uniform(shape, device, dtype, requires_grad=requires_grad)

	@staticmethod
	def full(shape: tuple, fill_value: Scalar, device: Device = Device.CPU,
			 dtype: DType = DType.FLOAT32, requires_grad: bool = False) -> Tensor:
		"""
		Creates a tensor with the specified shape filled with `fill_value`.

		Parameters
		----------
		shape : tuple
				The desired shape.
		fill_value : Scalar
				The value to fill the tensor with.
		device : str | Device | None
				Default device is CPU.
		dtype : str | DType | None
				Default dtype is float32.
		requires_grad : bool
				Default is False.

		Returns:
		   	A tensor filled with specfied values.
		"""
		return Tensor(
			data=Buffer.full(shape, fill_value=fill_value, device=device, dtype=dtype),
			requires_grad=requires_grad, device=device
		)

	@staticmethod
	def ones_like(shape: tuple, device: Device = Device.CPU,
			      dtype: DType = DType.FLOAT32, requires_grad: bool = False) -> Tensor:
		"""
		Creates a tensor with the specified shape filled with 1.0.

		Parameters
		----------
		shape : tuple
				The desired shape
		device : str | Device | None
				Default device is CPU
		dtype : str | DType | None
				Default dtype is float32
		requires_grad : bool
				Default is False

		Returns:
		    A tensor filled with 1.0.
		"""
		return Tensor.full(shape, 1.0, device, dtype, requires_grad)

	@staticmethod
	def zeros_like(shape: tuple, device: Device = Device.CPU,
			      dtype: DType = DType.FLOAT32, requires_grad: bool = False) -> Tensor:
		"""
		Creates a tensor with the specified shape filled with 0.0.

		Parameters
		----------
		shape : tuple
				The desired shape
		device : str | Device | None
				Default device is CPU
		dtype : str | DType | None
				Default dtype is float32
		requires_grad : bool
				Default is False

		Returns:
		    A tensor filled with 0.0.
		"""
		return Tensor.full(shape, 0.0, device, dtype, requires_grad)

	@staticmethod
	def add(x: Tensor, y: Tensor | Scalar) -> Tensor:
		"""
		Adds `x` and `y` tensors with broadcasting.

		Parameters
		----------
		x : Tensor
		y : Tensor

		Returns:
			The sum of `x` and `y`
		"""
		if isinstance(y, Scalar):
			y = Tensor(y, device=x.device)
		elif isinstance(x, Scalar):
			x = Tensor(x, device=y.device)
		return ops.Add.execute(x, y)

	@staticmethod
	def mul(x: Tensor | Tensor, y: Tensor | Scalar) -> Tensor:
		"""
		Multiplies `x` and `y` tensors with broadcasting.

		Parameters
		----------
		x : Tensor
		y : Tensor


		Returns:
			The product of `x` and `y`
		"""
		if isinstance(y, Scalar):
			y = Tensor(y, device=x.device)
		elif isinstance(x, Scalar):
			x = Tensor(x, device=y.device)
		return ops.Mul.execute(x, y)

	@staticmethod
	def sub(x: Tensor, y: Tensor | Scalar) -> Tensor:
		"""
		Subtracts `x` and `y` tensors with broadcasting.

		Parameters
		----------
		x : Tensor
		y : Tensor

		Returns:
			The difference of `x` and `y`
		"""
		if isinstance(y, Scalar):
			y = Tensor(y, device=x.device)
		elif isinstance(x, Scalar):
			x = Tensor(x, device=y.device)
		return ops.Sub.execute(x, y)

	@staticmethod
	def div(x: Tensor, y: Tensor | Scalar) -> Tensor:
		"""
		Divides `x` and `y` tensors with broadcasting.

		Parameters
		----------
		x : Tensor
		y :  Tensor

		Returns:
			The quotient of `x` and `y`
		"""
		if isinstance(y, Scalar):
			y = Tensor(y, device=x.device)
		elif isinstance(x, Scalar):
			x = Tensor(x, device=y.device)
		return ops.Div.execute(x, y)

	@staticmethod
	def matmul(x: Tensor, y: Tensor) -> Tensor:
		"""
		Matrix multiply `x` and `y` tensors.

		Parameters
		----------
		x : Tensor
		y : Tensor

		Returns:
			The matmul product of `x` and `y`
		"""
		return ops.MatMul.execute(x, y)

	@staticmethod
	def transpose(x: Tensor, *axes) -> Tensor:
		"""
		Transpose `x` tensor. Returns a new tensor.

		Parameters
		----------
		x : Tensor

		Returns:
			The transposed tensor
		"""
		if isinstance(axes[0], tuple):
			axes = axes[0]
		return ops.Transpose.execute(x, axes=axes)

	def sum(self, dim: int = -1, keepdim: bool = False) -> Tensor:
		"""
		Sum a tensor along dimension `dim`. keepdim is by default False.

		Parameters
		----------
		self : Tensor
		dim : int
			Dimension along which to sum, -1 means it sums all elements.

		Returns:
			A tensor.
		"""
		return ops.Sum.execute(self, dim=dim, keepdim=keepdim)

	def mean(self, dim: int = -1, keepdim: bool = False) -> Tensor:
		"""
		Find mean of a tensor along dimension `dim`. keepdim is by default False.

		Parameters
		----------
		self : Tensor
		dim : int
			Dimension along which to sum, -1 means it finds mean of all elements.

		Returns:
			A tensor.
		"""
		return ops.Mean.execute(self, dim=dim, keepdim=keepdim)

	def relu(self) -> Tensor:
		"""
		Applies ReLU activation to tensor.

		Parameters
		----------
		self : Tensor

		Returns:
			A tensor with ReLU activation applied
		"""
		return ops.Relu.execute(self)

	def leaky_relu(self, neg_slope: Scalar = 0.01) -> Tensor:
		"""
		Applies Leaky ReLU activation to tensor.

		Parameters
		----------
		self : Tensor

		Returns:
			A tensor with Leaky ReLU activation applied
		"""
		return ops.LeakyRelu.execute(self, neg_slope=neg_slope)

	def tanh(self) -> Tensor:
		"""
		Applies Tanh activation to tensor.

		Parameters
		----------
		self : Tensor

		Returns:
			A tensor with Tanh activation applied
		"""
		return ops.Tanh.execute(self)

	def sigmoid(self) -> Tensor:
		"""
		Applies Sigmoid activation to tensor.

		Parameters
		----------
		self : Tensor

		Returns:
			A tensor with Sigmoid activation applied
		"""
		return ops.Sigmoid.execute(self)

	def linear(self, weight: Tensor, bias: Tensor | None) -> Tensor:
		"""
		Applies a linear transformation to `self` using `weight` and `bias`.
		`self @ weight.T + bias`

		Parameters
		----------
		self : Tensor
		weight : Tensor
		bias : Tensor | None

		Returns:
			A tensor with linear transformation applied
		"""
		return self @ weight.T + bias if bias else self @ weight.T

	def max(self, dim: int = -1, keepdim: bool = False) -> Tensor:
		"""
		Find maximum of a tensor along dimension `dim`. keepdim is by default True.
		Which means the returned tensor shape is the same as the input tensor shape.

		Parameters
		----------
		self : Tensor
		dim : int
			Dimension along which to find the max, -1 means find max of full tensor.

		Returns:
			A tensor of the same shape as self.
		"""
		return ops.Max.execute(self, dim=dim, keepdim=keepdim)

	def exp(self) -> Tensor:
		"""
		Applies the exponential function to the tensor elementwise.

		Parameters
		----------
		self : Tensor

		Returns:
			A tensor of the same shape as self.
		"""
		return ops.Exp.execute(self)

	# TODO: Find which log
	def log(self) -> Tensor:
		"""
		Applies the logarithm function to the tensor elementwise.

		Parameters
		----------
		self : Tensor

		Returns:
			A tensor of the same shape as self.
		"""
		return ops.Log.execute(self)

	def sqrt(self) -> Tensor:
		"""
		Applies the sqaure root function to the tensor elementwise.

		Parameters
		----------
		self : Tensor

		Returns:
			A tensor of the same shape as self.
		"""
		return ops.Sqrt.execute(self)

	def log_softmax(self, dim: int = 1) -> Tensor:
		"""
		Applies the log-softmax function to the tensor along `dim`.

		Parameters
		----------
		self : Tensor
		dim : int
			Dimension along wich log-softmax should be applied

		Returns:
			A tensor of the same shape as self.
		"""
		t = self.max(dim=dim, keepdim=True)
		m = self - t
		e = m.exp()
		ss = e.sum(dim=dim, keepdim=True)
		return m - ss.log()

	def clamp(self, min: int | None = None, max: int | None = None) -> Tensor:
		return ops.Clamp.execute(self, min=min, max=max)

	def sequential(self, layers: list[callable[Tensor]]) -> None:
		return reduce(lambda inp, layer: layer(inp), layers, self)

	def squeeze(self, dim: list[int] | int) -> Tensor:
		return ops.Squeeze.execute(self, dim=dim)

	def unsqueeze(self, dim: list[int] | int) -> Tensor:
		return ops.Unsqueeze.execute(self, dim=dim)

	def cross_entropy_loss(self, target: Tensor) -> Tensor:
		"""
		Finds the cross-entropy loss between `self` and `target`.

		Parameters
		----------
		self : Tensor
		target: Tensor

		Returns:
			A tensor of shape (1, 1).
		"""
		if isinstance(target, Scalar):
			target = Tensor(target)
		return ops.CrossEntropy.execute(self, target)

	def bcewithlogitsloss(self, target: Tensor) -> Tensor:
		"""
		Finds the binary cross-entropy logits loss between `self` and `target`.

		Parameters
		----------
		self : Tensor
		target: Tensor

		Returns:
			A tensor of shape (1, 1).
		"""
		if isinstance(target, Scalar):
			target = Tensor(target)
		return ops.BCEWithLogitsLoss.execute(self, target)

	def argmax(self, axis: int = -1) -> Tensor:
		return Tensor(data=self.data.argmax(axis))

	def detach(self) -> Tensor:
		return Tensor(data=self.data, requires_grad=False, device=self.device)

	@staticmethod
	def where(cond: Tensor, inp: Tensor | Scalar, other: Tensor | Scalar) -> Tensor:
		return Tensor(data=cond.data.where(inp=inp if isinstance(inp, Scalar) else inp.data, other=other if isinstance(other, Scalar) else other.data), requires_grad=cond.requires_grad, device=cond.device)

	def backward(self) -> None:
		assert all(item == 1 for item in self.shape), "backward must be called on a scalar Tensor"

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

		self.grad = Tensor(1.0, device=self.device)

		# TODO: del _ctx
		for node in reversed(topo):
			if not node.grad:
				raise RuntimeError(f"Tensor {node} has no grad")
			upstream_grads: tuple[Buffer] = node._ctx.backward(node.grad.data)
			upstream_grads: list[Tensor] = [
				Tensor(g, requires_grad=False, device=g.device) for g in upstream_grads if g is not None
			]
			for p, g in zip(node._ctx.parents, upstream_grads):
				p.grad = g if not p.grad else p.grad + g

	def __getitem__(self, idx: slice) -> Tensor:
		"""
		dlgrad only supports slicing indexing.
		"""
		if isinstance(idx, slice) and idx.start is not None and idx.stop is not None and idx.step is None:
			s = idx.start*self.stride[0]
			ns = tuple([idx.stop-idx.start, *self.shape[1:]])

			buf = Buffer(data=self.data.ptr+s, shape=ns, device=self.device, dtype=self.dtype)

			return Tensor(data=buf, requires_grad=self.requires_grad, device=self.device)
		else:
			raise ValueError("dlgrad only supports slicing or start, stop are None")

	def __repr__(self) -> str:
		return f"Tensor<dtype: {self.dtype} device: {self.device}, shape: {self.shape}, ndim: {self.ndim}>"  # noqa: E501

	@property
	def T(self) -> Tensor:  # noqa: N802
		return Tensor.transpose(self, tuple([i for i in range(self.ndim)][::-1]))

	def __gt__(self, other: Scalar) -> Tensor:
		return Tensor(data=self.data>other)

	def __add__(self, other: Tensor | Scalar) -> Tensor:
		return Tensor.add(self, other)

	# TODO: Support rmul, lmul
	def __mul__(self: Tensor | Scalar, other: Tensor | Scalar) -> Tensor:
		return Tensor.mul(self, other)

	def __sub__(self, other: Tensor | Scalar) -> Tensor:
		return Tensor.sub(self, other)

	def __truediv__(self, other: Tensor | Scalar) -> Tensor:
		return Tensor.div(self, other)

	def __pow__(self, val: Scalar) -> Tensor:
		return Tensor(data=self.data**val, requires_grad=self.requires_grad)

	def __neg__(self) -> Tensor:
		return Tensor(data=-self.data, requires_grad=self.requires_grad)

	def __matmul__(self, other: Tensor) -> Tensor:
		return Tensor.matmul(self, other)

	def __eq__(self, other: Tensor) -> Tensor:
		return Tensor(data=self.data==other.data)

	def __hash__(self):  # noqa: ANN204
		return id(self)

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
- python slots
- kernel fusion, ex, relu after matmul
- functools cache in helpers
- profile memory, how many allocations and free, if any freed obj is being used
"""
