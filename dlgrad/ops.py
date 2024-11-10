from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dlgrad import Tensor

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import Scalar
from dlgrad.helpers import BinaryOps, BufferOps, check_broadcast, get_brodcast_tensor
from dlgrad.tensor import OP

# ------------ Buffer Ops -----------

def create_buffer_from_scalar(x: Scalar, device: Device) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.CREATE, device=device, x=x)

def uniform(shape: tuple, device: Device, **kwargs) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.UNIFORM, device=device, x=shape, **kwargs)

# ------------ Binary Ops -----------

class Add(OP):
    def forward(self, x: 'Tensor', y: 'Tensor') -> Buffer:
        x, y = get_brodcast_tensor(x, y)

        if check_broadcast(x.shape, y.shape):
            return dispatcher.dispatch(op=BinaryOps.ADD, device=x.device, x=x, y=y)
     
    def backward(self):
        pass

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