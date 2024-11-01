from dlgrad.buffer import Buffer
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import Scalar
from dlgrad.helpers import BufferOps, BinaryOps
from dlgrad.tensor import OP

# ------------ Buffer Ops -----------

def create_buffer_from_scalar(x: Scalar, **kwargs) -> Buffer:
    return dispatcher.dispatch(x=x, op=BufferOps.CREATE, device=kwargs.pop("device"), **kwargs)

def uniform(shape: tuple, **kwargs) -> Buffer:
    return dispatcher.dispatch(x=shape, op=BufferOps.UNIFORM, device=kwargs.pop("device"), **kwargs)

# ------------ Binary Ops -----------

class Add(OP):
    def forward(self, x, y):
        return dispatcher.dispatch(op=BinaryOps.ADD, device=x.device, x=x, y=y)
     
    def backward(self, *args, **kwargs):
        return super().backward(*args, **kwargs)
