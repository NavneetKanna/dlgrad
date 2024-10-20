from dlgrad.dispatch import dispatcher
from dlgrad.dtype import Scalar
from dlgrad.helpers import BufferOps
from dlgrad.tensor import OP
from dlgrad.buffer import Buffer


# ------------ Buffer Ops -----------

def create_buffer_from_scalar(x: Scalar, **kwargs) -> Buffer:
    return dispatcher.dispatch(x=x, op=BufferOps.CREATE, device=kwargs.pop("device"), **kwargs)

def uniform(shape: tuple, **kwargs) -> Buffer:
    return dispatcher.dispatch(shape, BufferOps.UNIFORM, kwargs.pop("device"), **kwargs)