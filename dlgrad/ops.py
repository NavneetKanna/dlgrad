
from dlgrad.buffer import Buffer
from dlgrad.dtype import Scalar
from dlgrad.helpers import CustomOps, check_broadcast, find_broadcast_dim
from dlgrad.tensor import OP

# ------------ Unary Ops -----------

# TODO: Remove compulsory tuple return type in backward
class Transpose(OP):
	def forward(self, x: Buffer, axes: tuple):  # noqa: ANN201
		return x.transpose(axes)

	def backward(self, upstream_grad: Buffer):  # noqa: ANN201
		return (upstream_grad.transpose(tuple([i for i in range(upstream_grad.ndim)][::-1])),)


# done
class Sum(OP):
	def forward(self, x: Buffer, dim: int = -1, keepdim: bool = False) -> Buffer:
		self.inp_shape = x.shape
		self.device = x.device
		self.dtype = x.dtype
		self.keepdim = keepdim
		self.dim = dim
		return x.sum(dim=dim, keepdim=keepdim)

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		t = Buffer.full(shape=self.inp_shape, fill_value=1.0, device=self.device, dtype=self.dtype)
		if not self.keepdim:
			upstream_grad.unsqueeze(self.dim)
		return (t*upstream_grad,)

# done
class Max(OP):
	def forward(self, x: Buffer, dim: int = -1, keepdim: bool = False) -> Buffer:
		self.inp_shape = x.shape
		self.device = x.device
		self.dim = dim
		self.x = x
		self.keepdim = keepdim
		self.out = x.max(dim=dim, keepdim=keepdim)
		return self.out

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		# TODO: Can we replace this with self.x == self.out ?
		max_with_1s = self.x.max(dim=self.dim, backward=True, out=self.out)
		if not self.keepdim:
			upstream_grad.unsqueeze(self.dim)

		return (max_with_1s*upstream_grad,)

# done
class Exp(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.out = x.exp()
		return self.out

	def backward(self, upstream_grad: Buffer) -> Buffer:
		return (upstream_grad * self.out,)


# done
class Log(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.x = x
		return x.log()

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		return (upstream_grad / self.x,)


# done
class Relu(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.out = x.relu()
		return self.out

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		return ((self.out>0.0) * upstream_grad,)


# done
class Sqrt(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.out = x.sqrt()
		return self.out

	def backward(self, grad_output: Buffer) -> tuple[Buffer]:
		return grad_output / (self.out*2)


# ------------ Binary Ops -----------

# TODO: Redundant calls to find_broadcast_dim
class Add(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x + y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
		return self.match_inp_shape(inp=self.x, upstream_grad=upstream_grad, dim=find_broadcast_dim(self.x.shape, self.y.shape)) if self.req_grad[0] else None, \
		  	   self.match_inp_shape(inp=self.y, upstream_grad=upstream_grad, dim=find_broadcast_dim(self.x.shape, self.y.shape)) if self.req_grad[1] else None  # noqa: E501


class Sub(OP):
	def forward(self, x: Buffer, y: Buffer | Scalar) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x - y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
		return self.match_inp_shape(inp=self.x, upstream_grad=upstream_grad, dim=find_broadcast_dim(self.x.shape, self.y.shape)) if self.req_grad[0] else None, \
		  	   self.match_inp_shape(inp=self.y, upstream_grad=-upstream_grad, dim=find_broadcast_dim(self.x.shape, self.y.shape)) if self.req_grad[1] else None  # noqa: E501


class Mul(OP):
	def forward(self, x: Buffer, y: Buffer | Scalar) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x*y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
		return self.match_inp_shape(inp=self.x, upstream_grad=upstream_grad*self.y, dim=find_broadcast_dim(self.x.shape, self.y.shape)) if self.req_grad[0] else None, \
		  	    self.match_inp_shape(inp=self.y, upstream_grad=upstream_grad*self.x, dim=find_broadcast_dim(self.x.shape, self.y.shape)) if self.req_grad[1] else None  # noqa: E501


class Div(OP):
	def forward(self, x: Buffer, y: Buffer | Scalar) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x/y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
		return self.match_inp_shape(inp=self.x, upstream_grad=upstream_grad/self.y, dim=find_broadcast_dim(self.x.shape, self.y.shape)) if self.req_grad[0] else None, \
		  	   self.match_inp_shape(inp=self.y, upstream_grad=(-upstream_grad*self.x)/self.y**2, dim=find_broadcast_dim(self.x.shape, self.y.shape)) if self.req_grad[1] else None  # noqa: E501


class MatMul(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		self.x = x
		self.y = y
		return x@y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		t1 = self.x.T
		t2 = self.y.T
		return (upstream_grad@t2, t1@upstream_grad)


# TODO: Fuse all ops performed here in C ?
class CrossEntropy(OP):
	def forward(self, logits: Buffer, target: Buffer, dim: int = 1) -> Buffer:
		assert logits.shape[0] == target.shape[0], f"logits shape[0] and target shape[0] does not match {logits.shape} != {target.shape}"  # noqa: E501

		self.target = target
		t = logits.max(dim=dim)
		m = logits - t
		e = m.exp()
		ss = e.sum(dim=dim)
		self.log_softmax_output = m - ss.log()
		tmp = Buffer.ce_forward(self.log_softmax_output, self.target)

		return -tmp.sum()

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		tmp = self.log_softmax_output.exp()
		Buffer.ce_backward(op=CustomOps.CE_BACKWARD, device=tmp.device, x=tmp, target=self.target)
		return (tmp,)
