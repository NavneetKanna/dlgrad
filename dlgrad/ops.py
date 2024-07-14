from dlgrad.dispatch import Dispatcher
from dlgrad.helpers import calculate_stride, BinaryOps, UnaryOps, BroadcastHelper, calculate_numel
from dlgrad.tensor import Tensor, TensorProperties
from dlgrad.runtime.cpu import CPU

class Op:
    """
    This class may not be required since we can store the parents in each op class, however,
    since parents variable is common to all ops, it is a good practice to have a super class
    and define it there (here).
    """
    def __init__(self) -> None:
        self.parents: tuple = None

class Broadcast(Op):
    def forward(self, x: Tensor, y: Tensor) -> tuple:
        shape1 = x.shape
        shape2 = y.shape

        if x.ndim > 2 or y.ndim > 2 and shape1 != shape2:
            print("Dlgrad does not support broadcasting for dims greater than 2")
        
        output_shape = []
        
        shape1 = shape1[::-1]
        shape2 = shape2[::-1]

        for i in range(max(len(shape1), len(shape2))):
            dim1 = shape1[i] if i < len(shape1) else 1
            dim2 = shape2[i] if i < len(shape2) else 1
            if dim1 == 1 or dim2 == 1 or dim1 == dim2:
                output_shape.append(max(dim1, dim2))
            else:
                # TODO: Add error here
                print("Shapes are not compatible for broadcasting")
        
        out_shape = tuple(output_shape[::-1])
        BroadcastHelper.out_len = calculate_numel(out_shape)
        
        y._ctx = self
        self.out_shape = out_shape
        # self.parents = (x, y)
        self.x, self.y = x, y

        return out_shape

    def backward(self):
        """
        Only applies to the 2nd inp, which is getting broadcasted
        """
        if self.x.shape[0] == self.y.shape[0]:
            "sum along axis0"
            tp = TensorProperties(view=False, offset=0, numel=calculate_numel(self.out_shape), shape=self.out_shape, ndim=1, stride=(1,), contig=True)
            out = Tensor(Dispatcher.dispatch(x=self.x, ops=UnaryOps.SUM, func=CPU.sum_axis0), device=self.x.device, dtype=self.x.dtype, properties=tp)
            
            self.y.grad += out

        elif self.x.shape[1] == self.y.shape[1]:
            "sum along axis1"
            tp = TensorProperties(view=False, offset=0, numel=calculate_numel(self.out_shape), shape=self.out_shape, ndim=1, stride=(1,), contig=True)
            out = Tensor(Dispatcher.dispatch(x=self.x, ops=UnaryOps.SUM, func=CPU._sum_axis1), device=self.x.device, dtype=self.x.dtype, properties=tp)
            
            self.y.grad += out
        else:
            "sum full Tensor"

class Add(Op):
    def forward(self, x: Tensor, y: Tensor, out_shape: tuple) -> Tensor:
        assert x.device == y.device, f"{x.device} and {y.device} does not match"

        tp = TensorProperties(view=False, offset=0, numel=calculate_numel(out_shape), shape=out_shape, ndim=len(out_shape), stride=calculate_stride(out_shape) if out_shape else (), contig=True)
        out = Tensor(Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.ADD), device=x.device, dtype=x.dtype, properties=tp)

        out._ctx = self
        self.parents = (x, y)
        self.x, self.y = x, y

        return out 

    def backward(self, grad_output):
        self.x.grad += 1.0 * grad_output
        self.y.grad += 1.0 * grad_output