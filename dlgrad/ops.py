from typing import TYPE_CHECKING, Optional

# if TYPE_CHECKING:
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

def uniform(shape: tuple, device: Device, **kwargs) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.UNIFORM, device=device, shape=shape, **kwargs)

def full(shape: tuple, fill_value: Scalar, device: Device) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.FULL, device=device, shape=shape, fill_value=fill_value)


# ------------ Unary Ops -----------

class Sum(OP):
    def forward(self, x: Buffer)-> Buffer:
        self.inp_shape = x.shape
        return dispatcher.dispatch(op=UnaryOps.SUM, device=x.device, x=x)
    
    def backward(self, upstream_grad: Buffer) -> Buffer:
        return dispatcher.dispatch(op=BufferOps.FULL, shape=self.inp_shape, fill_value=1.0) # * upstream_grad


# ------------ Binary Ops -----------

class Add(OP):
    def forward(self, x: Buffer, y: Buffer) -> Buffer:
        x, y = get_brodcast_tensor(x, y)

        if check_broadcast(x.shape, y.shape):
            return dispatcher.dispatch(op=BinaryOps.ADD, device=x.device, x=x, y=y)
     
    def backward(self, upstream_grad: Buffer) -> tuple[Optional[Buffer], Optional[Buffer]]:
        return upstream_grad if self.req_grad[0] else None, upstream_grad if self.req_grad[1] else None

class Neg(OP):
    def forward(self, x: Buffer) -> Buffer:
        return dispatcher.dispatch(op=BinaryOps.NEG, device=x.device, x=x)
    
    def backward(self):
        pass

class MatMul(OP):
    def forward(self, x: Buffer, y: Buffer):
        return dispatcher.dispatch(op=BinaryOps.MATMUL, device=x.device, x=x, y=y)

    def backward(self):
        pass