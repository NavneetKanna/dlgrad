from dlgrad.buffer import Buffer
from dlgrad.dtype import Scalar
from dlgrad.helpers import CustomOps, check_broadcast
from dlgrad.tensor import OP

# ------------ Unary Ops -----------

class Transpose(OP):
	def forward(self, x: Buffer):  # noqa: ANN201
		return x.transpose()

	def backward(self, upstream_grad: Buffer):  # noqa: ANN201
		return (upstream_grad.transpose(),)

# TODO: Maube add expand/broadcast_to instead of buffer.full in backward
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

class Mean(OP):
	def forward(self, x: Buffer, dim: int = -1, keepdim: bool = False) -> Buffer:
		self.inp_shape = x.shape
		self.device = x.device
		self.dtype = x.dtype
		self.keepdim = keepdim
		self.dim = dim
		if dim == -1:  # reduce over ALL elements
			self.N = x.numel
		else:
			self.N = x.shape[dim]
		return x.mean(dim=dim, keepdim=keepdim)

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		scale = 1.0 / self.N
		grad = Buffer.full(self.inp_shape, fill_value=scale, device=self.device, dtype=self.dtype)
		if not self.keepdim:
			upstream_grad.unsqueeze(self.dim)
			return (grad * upstream_grad,)

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

class Exp(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.out = x.exp()
		return self.out

	def backward(self, upstream_grad: Buffer) -> Buffer:
		return (upstream_grad * self.out,)


class Log(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.x = x
		return x.log()

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		return (upstream_grad / self.x,)


class Relu(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.out = x.relu()
		return self.out

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		return ((self.out.where(inp=1.0, other=0.0)) * upstream_grad,)


class Sigmoid(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.out = x.sigmoid()
		return self.out

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		return ((self.out * -(self.out - 1.0)) * upstream_grad,)


class LeakyRelu(OP):
	def forward(self, x: Buffer, neg_slope: Scalar = 0.01) -> Buffer:
		self.neg_slope = neg_slope
		self.out = x.leaky_relu(neg_slope=neg_slope)
		return self.out

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		return ((self.out.where(inp=1.0, other=self.neg_slope)) * upstream_grad,)


class Tanh(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.out = x.tanh()
		return self.out

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		# TODO: Remove buffer.full() and add support for scalar in buffer sub
		return ((Buffer.full(self.out.shape, 1.0, self.out.device, self.out.dtype) - self.out**2) * upstream_grad,)

class Sqrt(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.out = x.sqrt()
		return self.out

	def backward(self, grad_output: Buffer) -> tuple[Buffer]:
		return (grad_output / (self.out*2),)

class Clamp(OP):
	def forward(self, x: Buffer, min: int | None, max: int | None) -> Buffer:
		self.min = min
		self.max = max
		self.out = x.clamp(min, max)
		return self.out

	def backward(self, grad_output: Buffer) -> tuple[Buffer]:
		return (grad_output.clamp(self.min, self.max))

class Squeeze(OP):
	def forward(self, x: Buffer, dim: list[int] | int) -> Buffer:
		self.dim = dim
		x.squeeze(dim)
		return x

	def backward(self, grad_output: Buffer) -> tuple[Buffer]:
		grad_output.unsqueeze(self.dim)
		return (grad_output,)

class Unsqueeze(OP):
	def forward(self, x: Buffer, dim: list[int] | int) -> Buffer:
		self.dim = dim
		x.unsqueeze(dim)
		return x

	def backward(self, grad_output: Buffer) -> tuple[Buffer]:
		grad_output.squeeze(self.dim)
		return (grad_output,)

# ------------ Binary Ops -----------

class Add(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x + y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
		grad_x = self.reduce_grad_for_broadcasting(upstream_grad, self.x.shape) if self.req_grad[0] else None
		grad_y = self.reduce_grad_for_broadcasting(upstream_grad, self.y.shape) if self.req_grad[1] else None

		return grad_x, grad_y


class Sub(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x - y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
		grad_x = self.reduce_grad_for_broadcasting(upstream_grad, self.x.shape) if self.req_grad[0] else None
		grad_y = self.reduce_grad_for_broadcasting(-upstream_grad, self.y.shape) if self.req_grad[1] else None

		return grad_x, grad_y


class Mul(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x*y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
		grad_x = self.reduce_grad_for_broadcasting(upstream_grad*self.y, self.x.shape) if self.req_grad[0] else None
		grad_y = self.reduce_grad_for_broadcasting(upstream_grad*self.x, self.y.shape) if self.req_grad[1] else None

		return grad_x, grad_y


class Div(OP):
	def forward(self, x: Buffer, y: Buffer | Scalar) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x/y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
		grad_x = self.reduce_grad_for_broadcasting(upstream_grad/self.y, self.x.shape) if self.req_grad[0] else None
		grad_y = self.reduce_grad_for_broadcasting((-upstream_grad*self.x)/self.y**2, self.y.shape) if self.req_grad[1] else None

		return grad_x, grad_y


class MatMul(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		self.x = x
		self.y = y

		return x@y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		t1 = self.x.T
		t2 = self.y.T

		return (upstream_grad@t2, t1@upstream_grad)


class CrossEntropy(OP):
	def forward(self, logits: Buffer, target: Buffer, dim: int = 1) -> Buffer:
		assert logits.shape[0] == target.shape[0], f"logits shape[0] and target shape[0] does not match {logits.shape} != {target.shape}"  # noqa: E501

		self.target = target
		t = logits.max(dim=dim, keepdim=True)
		m = logits - t
		e = m.exp()
		ss = e.sum(dim=dim, keepdim=True)
		self.log_softmax_output = m - ss.log()
		tmp = Buffer.ce_forward(self.log_softmax_output, self.target)

		return -tmp.sum()

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		tmp = self.log_softmax_output.exp()
		Buffer.ce_backward(op=CustomOps.CE_BACKWARD, device=tmp.device, x=tmp, target=self.target)
		return (tmp,)

# https://publish.obsidian.md/kamilelukosiute/pytorch/What+does+BCEWithLogits+actually+do%3F
class BCEWithLogitsLoss(OP):
	def forward(self, probs: Buffer, target: Buffer) -> Buffer:
		assert probs.shape == target.shape, f"shape mismatch {probs.shape} vs {target.shape}"

		max_val = (-probs).clamp(min=0) # or (-probs).relu()
		self.probs = probs
		self.target = target
		self.loss = ((1.0-target) * probs) + max_val + ((-max_val).exp() + (-(probs+max_val)).exp()).log()
		return self.loss.mean()

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		N = self.probs.numel
		grad = ((self.probs.sigmoid() - self.target) * upstream_grad) / float(N)
		return (grad,)
