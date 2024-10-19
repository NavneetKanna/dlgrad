from dlgrad.dtype import Scalar
from dlgrad.tensor import OP
from dlgrad.dispatch import dispatcher
from enum import Enum, auto
from dlgrad.helpers import BufferOps



# ------------ Buffer Ops -----------
def create_buffer_from_scalar(x: Scalar, **kwargs):
    return dispatcher.dispatch(x, BufferOps.CREATE, **kwargs)
