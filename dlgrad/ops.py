from typing import Optional

from dlgrad.buffer import Buffer
from dlgrad.helpers import check_broadcast 
from dlgrad.tensor import OP

# ------------ Unary Ops -----------

def transpose(x: Buffer):
    return x.transpose()

class Sum(OP):
    def forward(self, x: Buffer, dim: int)-> Buffer:
        self.inp_shape = x.shape
        return x.sum(dim=dim)
    
    def backward(self, upstream_grad: Buffer) -> Buffer:
        pass
        # return dispatcher.dispatch(op=BufferOps.FULL, shape=self.inp_shape, fill_value=1.0) # * upstream_grad


# ------------ Binary Ops -----------

class Add(OP):
    def forward(self, x: Buffer, y: Buffer) -> Buffer:
        # x, y = get_brodcast_tensor(x, y)

        if check_broadcast(x.shape, y.shape):
            return x+y
     
    def backward(self, upstream_grad: Buffer) -> tuple[Optional[Buffer], Optional[Buffer]]:
        if self.req_grad[0]:
            if upstream_grad.shape == self.parents[0].shape:
                pass
            else:
                pass

        return upstream_grad if self.req_grad[0] else None, upstream_grad if self.req_grad[1] else None

class Sub(OP):
    def forward(self, x: Buffer, y: Buffer) -> Buffer:
        # x, y = get_brodcast_tensor(x, y)

        if check_broadcast(x.shape, y.shape):
            return x-y
     
    def backward(self, upstream_grad: Buffer) -> tuple[Optional[Buffer], Optional[Buffer]]:
        return ...

class Mul(OP):
    def forward(self):
        return ...
    
    def backward(self):
        return ...

class Neg(OP):
    def forward(self, x: Buffer) -> Buffer:
        return x.neg()
    
    def backward(self):
        pass

class MatMul(OP):
    def forward(self, x: Buffer, y: Buffer):
        return x.matmul(y)

    def backward(self):
        pass