from dlgrad.buffer import Buffer
from dlgrad.helpers import check_broadcast
from dlgrad.tensor import OP

# ------------ Unary Ops -----------


def transpose(x: Buffer) -> Buffer:
	return x.transpose()


class Sum(OP):
	def forward(self, x: Buffer, dim: int | None) -> Buffer:
		self.inp_shape = x.shape
		self.device = x.device
		return x.sum(dim=dim)

	def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
		return (Buffer.full(shape=self.inp_shape, fill_value=1.0, device=self.device),)


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


class Neg(OP):
	def forward(self, x: Buffer) -> Buffer:
		return x.neg()

	def backward(self):  # noqa: ANN201
		pass


class MatMul(OP):
	def forward(self, x: Buffer, y: Buffer) -> Buffer:
		return x.matmul(y)

	def backward(self):  # noqa: ANN201
		pass

class Relu(OP):
	def forward(self, x: Buffer) -> Buffer:
		self.out = x.relu()
		return self.out

	def backward(self, upstream_grad: Buffer) -> Buffer:
		return ((self.out>0.0) * upstream_grad,)
