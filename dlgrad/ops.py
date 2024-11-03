from dlgrad.buffer import Buffer
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import Scalar
from dlgrad.helpers import BinaryOps, BufferOps
from dlgrad.tensor import OP

# ------------ Buffer Ops -----------

def create_buffer_from_scalar(x: Scalar, **kwargs) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.CREATE, device=kwargs.pop("device"), x=x, **kwargs)

def uniform(shape: tuple, **kwargs) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.UNIFORM, device=kwargs.pop("device"), x=shape, **kwargs)

def arange(shape: tuple, **kwargs) -> Buffer:
    return dispatcher.dispatch(op=BufferOps.ARANGE, device=kwargs.pop("device"), x=shape, **kwargs)

# ------------ Binary Ops -----------

class Add(OP):
    def forward(self, x, y) -> Buffer:
        if y.ndim > x.ndim:
            x, y = y, x

        return dispatcher.dispatch(op=BinaryOps.ADD, device=x.device, x=x, y=y)
     
    def backward(self, *args, **kwargs):
        return super().backward(*args, **kwargs)
