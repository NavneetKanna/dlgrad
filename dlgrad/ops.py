from enum import Enum, auto

from dlgrad.dispatch import dispatcher
from dlgrad.dtype import Scalar
from dlgrad.helpers import BufferOps
from dlgrad.tensor import OP


# ------------ Buffer Ops -----------

def create_buffer_from_scalar(x: Scalar, **kwargs):
    return dispatcher.dispatch(x, BufferOps.CREATE, **kwargs)
