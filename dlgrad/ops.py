from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dlgrad import Tensor

from dlgrad.buffer import Buffer
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import Scalar
from dlgrad.helpers import BinaryOps, BufferOps
from dlgrad.tensor import OP
from dlgrad.device import Device


# ------------ Buffer Ops -----------

def create_buffer_from_scalar(x: Scalar, device: Device) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.CREATE, device=device, x=x)

def uniform(shape: tuple, device: Device) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.UNIFORM, device=device, x=shape)

# ------------ Binary Ops -----------

class Add(OP):
    def forward(self, x: 'Tensor', y: 'Tensor') -> Buffer:
        if y.ndim > x.ndim:
            x, y = y, x

        return dispatcher.dispatch(op=BinaryOps.ADD, device=x.device, x=x, y=y)
     
    def backward(self):
        pass

class Neg(OP):
    def forward(self, x: 'Tensor') -> Buffer:
        return dispatcher.dispatch(op=BinaryOps.NEG, device=x.device, x=x)