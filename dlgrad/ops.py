import dlgrad.graph as graph
from dlgrad.dispatch import Dispatcher
from dlgrad.helpers import (BinaryOps, BroadcastHelper, UnaryOps,
                            calculate_numel, calculate_stride, get_graph)
from dlgrad.runtime.cpu import CPU
from dlgrad.tensor import Tensor, TensorProperties


class Op:
    """
    This class may not be required since we can store the parents in each op class, however,
    since parents variable is common to all ops, it is a good practice to have a super class
    and define it there (here).
    """
    def __init__(self) -> None:
        self.parents: tuple = ()

# TODO: Add graph 
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
        self.out_shape = y.shape
        # self.parents = (x, y)
        self.x, self.y = x, y

        tp = TensorProperties(view=None, offset=None, numel=None, shape=None, ndim=None, stride=None, contig=None, metadata={'created_by': None, 'ops': None})
        out = Tensor(data=None, requires_grad=None, device=None, dtype=None, properties=tp)

        if get_graph():
            graph.add_edge(child=out, parents=(y,))

        return out_shape

    def backward(self, grad_output):
        """
        Only applies to the 2nd inp, which is getting broadcasted
        """
        if self.x.shape[0] == self.y.shape[0]:
            tp = TensorProperties(view=False, offset=0, numel=calculate_numel(self.out_shape), shape=self.out_shape, ndim=1, stride=(1,), contig=True)
            out = Tensor(Dispatcher.dispatch(x=grad_output, ops=UnaryOps.SUM, func=CPU._sum_axis1), device=self.x.device, dtype=self.x.dtype, properties=tp)
            
            self.y.grad = out 
        elif self.x.shape[1] == self.y.shape[1]:
            tp = TensorProperties(view=False, offset=0, numel=calculate_numel(self.out_shape), shape=self.out_shape, ndim=1, stride=(1,), contig=True)
            out = Tensor(Dispatcher.dispatch(x=grad_output, ops=UnaryOps.SUM, func=CPU.sum_axis0), device=self.x.device, dtype=self.x.dtype, properties=tp)

            self.y.grad = out 

class Add(Op):
    def forward(self, x: Tensor, y: Tensor, out_shape: tuple) -> Tensor:
        assert x.device == y.device, f"{x.device} and {y.device} does not match"

        tp = TensorProperties(view=False, offset=0, numel=calculate_numel(out_shape), shape=out_shape, ndim=len(out_shape), stride=calculate_stride(out_shape) if out_shape else (), contig=True, metadata={'created_by': 'Add', 'ops': BinaryOps})
        out = Tensor(Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.ADD), device=x.device, dtype=x.dtype, properties=tp)

        out._ctx = self
        self.parents = (x, y)
        self.x, self.y = x, y

        if get_graph():
            graph.add_edge(child=out, parents=(x, y))

        return out 

    def backward(self, grad_output):
        self.x.grad = grad_output if self.x.grad is None else self.x.grad + grad_output
        self.y.grad = grad_output if self.y.grad is None else self.y.grad + grad_output

class Sum(Op):
    def forward(self, x: Tensor, axis: int = None):
        if axis == 0:
            out_shape = x.shape[1]
            tp = TensorProperties(view=False, offset=0, numel=calculate_numel(out_shape), shape=out_shape, ndim=1, stride=(1,), contig=True, metadata={'created_by': 'Sum', 'ops': UnaryOps})
            out = Tensor(Dispatcher.dispatch(x=x, ops=UnaryOps.SUM, func=CPU.sum_axis0), device=x.device, dtype=x.dtype, properties=tp)

            if get_graph():
                graph.add_edge(child=out, parents=(x,))

            return out
        elif axis == 1:
            out_shape = x.shape[1]
            tp = TensorProperties(view=False, offset=0, numel=calculate_numel(out_shape), shape=out_shape, ndim=1, stride=(1,), contig=True, metadata={'created_by': 'Sum', 'ops': UnaryOps})
            out = Tensor(Dispatcher.dispatch(x=x, ops=UnaryOps.SUM, func=CPU._sum_axis1), device=x.device, dtype=x.dtype, properties=tp)
            
            if get_graph():
                graph.add_edge(child=out, parents=(x,))

            return out
        else:
            out_shape = ()
            tp = TensorProperties(view=False, offset=0, numel=1, shape=out_shape, ndim=1, stride=(1,), contig=True, metadata={'created_by': 'Sum', 'ops': UnaryOps})
            out = Tensor(Dispatcher.dispatch(x=x, ops=UnaryOps.SUM, func=CPU.sum), device=x.device, dtype=x.dtype, properties=tp)

            self.x = x
            out._ctx = self
            self.parents = (x,)
            
            if get_graph():
                graph.add_edge(child=out, parents=(x,))

            return out

    def backward(self, grad_output):
        # NOTE: backward only works for axis=None 
        self.x.grad = Tensor.ones(self.x.shape) if self.x.grad is None else self.x.grad + Tensor.ones(self.x.shape)
