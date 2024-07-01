from dlgrad.tensor import Tensor, TensorProperties
from dlgrad.dispatch import Dispatcher
from dlgrad.helpers import calculate_stride, BinaryOps, BroadcastHelper


class Op:
    """
    This class may not be required since we can store the parents in each op class, however,
    since parents variable is common to all ops, it is a good practice to have a super class
    and define it there (here).
    """
    def __init__(self) -> None:
        self.parents: tuple = None

class Add(Op):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out_shape = Tensor._broadcast(x, y)
        out_len = 1
        for i in out_shape:
            out_len *= i

        # TODO: Remove this in future
        BroadcastHelper.out_len = out_len

        tp = TensorProperties(view=False, offset=0, numel=out_len, shape=out_shape, ndim=len(out_shape), stride=calculate_stride(out_shape) if out_shape else (), contig=True)
        out = Tensor(Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.ADD), device=x.device, dtype=x.dtype, properties=tp)

        def _backward(): 
            x.grad = out.grad
            y.grad = out.grad

        out._ctx = self
        self.parents = (x, y)

        return out 

    def backward(self):
        pass