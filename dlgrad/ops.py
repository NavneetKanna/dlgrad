from dlgrad.buffer import Buffer
from dlgrad.helpers import CustomOps, check_broadcast
from dlgrad.tensor import OP

# ------------ Unary Ops -----------

def transpose(x: Buffer) -> Buffer:
	return x.transpose()


class Sum(OP):
	def forward(self, x: Buffer, dim: int = -1) -> Buffer:
		self.inp_shape = x.shape
		self.device = x.device
		return x.sum(dim=dim)

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		return (Buffer.full(shape=self.inp_shape, fill_value=1.0, device=self.device),)


class Max(OP):
	def forward(self, x: Buffer, dim: int = -1) -> Buffer:
		self.inp_shape = x.shape
		self.device = x.device
		self.x = x
		self.out, self.max_with_1s = x.max(dim=dim)
		return self.out

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		return (self.max_with_1s,)


class Exp(OP):
	def forward(self, x: Buffer) -> Buffer:
		return x.exp()

	def backward(self, upstream_grad: Buffer) -> Buffer:
		pass


class Log(OP):
	def forward(self, x: Buffer) -> Buffer:
		return x.log()

	def backward(self, upstream_grad: Buffer) -> Buffer:
		pass


class Relu(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.out = x.relu()
		return self.out

	def backward(self, upstream_grad: Buffer) -> Buffer:
		return ((self.out>0.0) * upstream_grad,)


# ------------ Binary Ops -----------

class Add(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x + y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
		return self.match_inp_shape(inp=self.x, upstream_grad=upstream_grad) if self.req_grad[0] else None, \
		  	   self.match_inp_shape(inp=self.y, upstream_grad=upstream_grad) if self.req_grad[1] else None


class Sub(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x - y

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
		return self.match_inp_shape(inp=self.x, upstream_grad=upstream_grad) if self.req_grad[0] else None, \
		  	   self.match_inp_shape(inp=self.y, upstream_grad=-upstream_grad) if self.req_grad[1] else None


class Mul(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		self.x = x
		self.y = y
		if check_broadcast(x.shape, y.shape):
			return x*y

	def backward(self, upstream_grad: Buffer) -> Buffer:
		return self.match_inp_shape(inp=self.x, upstream_grad=upstream_grad) * self.x if self.req_grad[0] else None, \
		  	   self.match_inp_shape(inp=self.y, upstream_grad=-upstream_grad) * self.y if self.req_grad[1] else None


class MatMul(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		return x.matmul(y)

	def backward(self):  # noqa: ANN201
		pass


class CrossEntropy(OP):
	def forward(self, logits: Buffer, target: Buffer, dim: int = 1) -> Buffer:
		self.target = target
		t, _ = logits.max(dim=dim)
		m = logits - t
		e = m.exp()
		ss = e.sum(dim=dim)
		self.log_softmax_output = m - ss.log()
		tmp = Buffer.ce_forward(self.log_softmax_output, self.target)
		return -tmp.sum()

	def backward(self, upstream_grad: Buffer) -> Buffer:
		tmp = self.log_softmax_output.exp()
		Buffer.ce_backward(op=CustomOps.CE_BACKWARD, device=tmp.device, x=tmp, target=self.target)
		return (tmp,)


class LogSoftmax(OP):
	def forward(self, x: Buffer) -> Buffer:
		m = x - x.max()
		e = m.exp()
		ss = e.sum()
		self.out = m - ss.log()
		return self.out

	def backward(self, upstream_grad: Buffer) -> Buffer:
		softmax_output = self.out.exp()
		return upstream_grad - softmax_output * upstream_grad.sum(1)
