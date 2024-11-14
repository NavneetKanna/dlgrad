from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dlgrad import Tensor

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import Scalar
from dlgrad.helpers import (BinaryOps, BufferOps, UnaryOps, check_broadcast,
                            get_brodcast_tensor)
from dlgrad.tensor import OP

# ------------ Buffer Ops -----------

def create_buffer_from_scalar(x: Scalar, device: Device) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.CREATE, device=device, x=x)

def uniform(shape: tuple|tuple, device: Device, **kwargs) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.UNIFORM, device=device, x=shape, **kwargs)


# ------------ Unary Ops -----------

class Sum(OP):
    def forward(self, x: 'Tensor')-> Buffer:
        return dispatcher.dispatch(op=UnaryOps.SUM, device=x.device, x=x)
    
    def backward(self, upstream_grad: 'Tensor'):
        pass


# ------------ Binary Ops -----------

class Add(OP):
    def forward(self, x: 'Tensor', y: 'Tensor') -> Buffer:
        x, y = get_brodcast_tensor(x, y)

        if check_broadcast(x.shape, y.shape):
            return dispatcher.dispatch(op=BinaryOps.ADD, device=x.device, x=x, y=y)
     
    def backward(self, upstream_grad: 'Tensor') -> tuple[Optional['Tensor'], Optional['Tensor']]:
        return upstream_grad if self.req_grad[0] else None, upstream_grad if self.req_grad[1] else None

class Neg(OP):
    def forward(self, x: 'Tensor') -> Buffer:
        return dispatcher.dispatch(op=BinaryOps.NEG, device=x.device, x=x)
    
    def backward(self):
        pass

class MatMul(OP):
    def forward(self, x: 'Tensor', y: 'Tensor'):
        return dispatcher.dispatch(op=BinaryOps.MATMUL, device=x.device, x=x, y=y)

    def backward(self):
        pass