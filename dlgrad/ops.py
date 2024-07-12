from dlgrad.dispatch import Dispatcher
from dlgrad.helpers import calculate_stride, BinaryOps, BroadcastHelper, calculate_numel
from dlgrad.tensor import Tensor, TensorProperties

class Op:
    """
    This class may not be required since we can store the parents in each op class, however,
    since parents variable is common to all ops, it is a good practice to have a super class
    and define it there (here).
    """
    def __init__(self) -> None:
        self.parents: tuple = None

class Broadcast(Op):
    def forward(self, x: Tensor, y: Tensor):
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

    def backward(self):
        """
        Only applies to the 2nd inp, which is getting broadcasted
        """
        pass

class Add(Op):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out_shape = Tensor._broadcast(x, y)

        # TODO: Remove this in future
        BroadcastHelper.out_len = calculate_numel(out_shape)

        tp = TensorProperties(view=False, offset=0, numel=calculate_numel(out_shape), shape=out_shape, ndim=len(out_shape), stride=calculate_stride(out_shape) if out_shape else (), contig=True)
        out = Tensor(Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.ADD), device=x.device, dtype=x.dtype, properties=tp)

        def _backward(): 
            x.grad = out.grad
            y.grad = out.grad

        out._ctx = self
        self.parents = (x, y)
        self.x, self.y = x, y

        return out 

    def backward(self, grad_output):
        self.x.grad += 1.0 * grad_output
        self.y.grad += 1.0 * grad_output