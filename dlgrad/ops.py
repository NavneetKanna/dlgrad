
from dlgrad.buffer import Buffer
from dlgrad.helpers import check_broadcast, get_sum_over_dims
from dlgrad.tensor import OP

# ------------ Unary Ops -----------

def transpose(x: Buffer) -> Buffer:
    return x.transpose()

class Sum(OP):
    def forward(self, x: Buffer, dim: int | None)-> Buffer:
        self.inp_shape = x.shape
        self.device = x.device
        return x.sum(dim=dim)

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        print("sum backward called")
        return (Buffer.full(shape=self.inp_shape, fill_value=1.0, device=self.device),)


# ------------ Binary Ops -----------

class Add(OP):
    def forward(self, x: Buffer, y: Buffer) -> Buffer:
        self.x = x
        self.y = y
        if check_broadcast(x.shape, y.shape):
            return x+y

    def match_inp_shape(self, inp_shape: tuple, upstream_grad: Buffer) -> Buffer:
        dims = get_sum_over_dims(inp_shape=inp_shape, grad_shape=upstream_grad.shape)
        if not dims:
            return upstream_grad

        for i in dims:
            upstream_grad = upstream_grad.sum(dim=i)

        return upstream_grad

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
        print("add backward called")
        return self.match_inp_shape(inp_shape=self.x.shape, upstream_grad=upstream_grad) if self.req_grad[0] else None, self.match_inp_shape(inp_shape=self.y.shape, upstream_grad=upstream_grad) if self.req_grad[1] else None

class Sub(OP):
    def forward(self, x: Buffer, y: Buffer) -> Buffer:
        if check_broadcast(x.shape, y.shape):
            return x-y

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
        return ...

class Mul(OP):
    def forward(self):  # noqa: ANN201
        return ...

    def backward(self):  # noqa: ANN201
        return ...

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
