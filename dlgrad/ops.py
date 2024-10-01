from dlgrad import graph
from dlgrad.dispatch import Dispatcher
from dlgrad.helpers import (BinaryOps, CustomOps, UnaryOps, calculate_numel,
                            calculate_stride, calculate_sum_axis,
                            calculate_uops, get_broadcast_shape, get_graph)
from dlgrad.tensor import Tensor, TensorProperties


class Op:
    """
    This class may not be required since we can store the parents in each op class, however,
    since parents variable is common to all ops, it is a good practice to have a super class
    and define it there (here).
    """
    def __init__(self) -> None:
        self.parents: tuple = ()

class Broadcast(Op):
    def forward(self, x: Tensor, y: Tensor) -> tuple:
        if x.shape == y.shape:
            return x.shape

        out_shape = get_broadcast_shape(x, y) 

        y._ctx = self
        self.out_shape = y.shape
        # self.parents = (x, y)
        self.x, self.y = x, y

        if get_graph():
            # TODO: Fix this
            tp = TensorProperties(
                view=None, offset=None, numel=None, shape=x.shape,
                ndim=None, stride=None, contig=None, metadata={"created_by": "broadcast_graph", "ops": None},
            )
            out = Tensor(data=None, requires_grad=None, device=None, dtype=None, properties=tp)
            graph.add_edge(child=out, parents=(y,))

        return out_shape

    def backward(self, grad_output):
        """
        Only applies to the 2nd inp, which is getting broadcasted
        """
        axis = calculate_sum_axis(self.x.shape, self.y.shape)
        tp = TensorProperties(
            view=False, offset=0, numel=calculate_numel(self.out_shape), shape=self.out_shape, 
            ndim=1, stride=(1,), contig=True,
        )
        out = Tensor(
            Dispatcher.dispatch(x=grad_output, ops=UnaryOps.SUM, axis=axis),
            device=self.x.device, dtype=self.x.dtype, properties=tp,
        )

        self.y.grad = out

class Add(Op):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.device == y.device, f"{x.device} and {y.device} does not match"

        out_shape = Broadcast().forward(x, y)
        print("out shape ", out_shape)

        tp = TensorProperties(
            view=False, offset=0, numel=calculate_numel(out_shape), shape=out_shape,
            ndim=len(out_shape), stride=calculate_stride(out_shape) if out_shape else (), contig=True, 
            metadata={"created_by": "Add", "ops": "BinaryOps"}
        )
        out = Tensor(
            Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.ADD),
            device=x.device, dtype=x.dtype, properties=tp
        )

        out._ctx = self
        self.parents = (x, y)
        self.x, self.y = x, y

        if get_graph():
            graph.add_edge(child=out, parents=(x, y))

        return out

    def backward(self, grad_output):
        self.x.grad = grad_output if self.x.grad is None else self.x.grad + grad_output
        self.y.grad = grad_output if self.y.grad is None else self.y.grad + grad_output


class Div(Op):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.device == y.device, f"{x.device} and {y.device} does not match"
        out_shape = Broadcast().forward(x, y)
        tp = TensorProperties(
            view=False, offset=0, numel=calculate_numel(out_shape), shape=out_shape,
            ndim=len(out_shape), stride=calculate_stride(out_shape) if out_shape else (), contig=True, 
            metadata={"created_by": "Div", "ops": "BinaryOps"}
        )
        out = Tensor(
            Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.DIV),
            device=x.device, dtype=x.dtype, properties=tp
        )

        out._ctx = self
        self.parents = (x, y)
        self.x, self.y = x, y

        if get_graph():
            graph.add_edge(child=out, parents=(x, y))

        return out

    def backward(self, grad_output):
        pass


class Sub(Op):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.device == y.device, f"{x.device} and {y.device} does not match"

        out_shape = Broadcast().forward(x, y)
        tp = TensorProperties(
            view=False, offset=0, numel=calculate_numel(out_shape), shape=out_shape,
            ndim=len(out_shape), stride=calculate_stride(out_shape) if out_shape else (), contig=True, 
            metadata={"created_by": "Sub", "ops": "BinaryOps"}
        )
        out = Tensor(
            Dispatcher.dispatch(x=x, y=y, ops=BinaryOps.SUB),
            device=x.device, dtype=x.dtype, properties=tp
        )

        out._ctx = self
        self.parents = (x, y)
        self.x, self.y = x, y

        if get_graph():
            graph.add_edge(child=out, parents=(x, y))

        return out

    def backward(self, grad_output):
        pass


class Sum(Op):
    def forward(self, x: Tensor, axis=None, keepdim=False):
        out_shape, numel, ndim, stride = calculate_uops(x.shape, axis, keepdim)
        tp = TensorProperties(
            view=False, offset=0, numel=numel, shape=out_shape,
            ndim=ndim, stride=stride, contig=True, metadata={"created_by": "Sum", "ops": "UnaryOps"},
        )
        out = Tensor(
            Dispatcher.dispatch(x=x, ops=UnaryOps.SUM, axis=axis),
            device=x.device, dtype=x.dtype, properties=tp
        )

        if get_graph():
            graph.add_edge(child=out, parents=(x,))

        self.x = x
        out._ctx = self
        self.parents = (x,)

        return out

    def backward(self, grad_output):
        print("sum backward")
        # NOTE: backward only works for axis=None:
        self.x.grad = (
            Tensor.ones(self.x.shape)
            if self.x.grad is None
            else self.x.grad + Tensor.ones(self.x.shape)
        )


class Log(Op):
    def forward(self, x: Tensor):
        tp = TensorProperties(
            view=False, offset=0, numel=x.numel, shape=x.shape,
            ndim=x.ndim, stride=x.stride, contig=True, metadata={"created_by": "Log", "ops": "UnaryOps"},
        )
        out = Tensor(
            Dispatcher.dispatch(x=x, ops=UnaryOps.LOG),
            device=x.device, dtype=x.dtype, properties=tp
        )

        if get_graph():
            graph.add_edge(child=out, parents=(x,))

        self.x = x
        out._ctx = self
        self.parents = (x,)

        return out

    def backward(self, grad_output):
        self.x.grad = grad_output / self.x


class Max(Op):
    def forward(self, x: Tensor, axis=None, keepdim=False):
        out_shape, numel, ndim, stride = calculate_uops(x.shape, axis, keepdim)
        tp = TensorProperties(
            view=False, offset=0, numel=numel, shape=out_shape,
            ndim=ndim, stride=stride, contig=True, metadata={"created_by": "Max", "ops": "UnaryOps"},
        )
        out = Tensor(
            Dispatcher.dispatch(x=x, ops=UnaryOps.MAX, func="max"),
            device=x.device, dtype=x.dtype, properties=tp
        )

        if get_graph():
            graph.add_edge(child=out, parents=(x,))

        self.x = x
        self.out = out
        out._ctx = self
        self.parents = (x,)

        # print("x")
        # print(x.numpy())
        # print("max out")
        # print(out.numpy())
        return out

    def backward(self, grad_output):
        # NOTE: backward only works for axis=None:
        self.x.grad = self.x == self.out


class Relu(Op):
    def forward(self, x: Tensor):
        tp = TensorProperties(
            view=False, offset=0, numel=x.numel, shape=x.shape,
            ndim=x.ndim, stride=x.stride, contig=True, metadata={"created_by": "Relu", "ops": "UnaryOps"},
        )
        out = Tensor(
            Dispatcher.dispatch(x=x, ops=UnaryOps.MAX, func="relu"),
            device=x.device, dtype=x.dtype, properties=tp
        )
        
        if get_graph():
            graph.add_edge(child=out, parents=(x,))

        return out


class Exp(Op):
    def forward(self, x: Tensor):
        tp = TensorProperties(
            view=False, offset=0, numel=x.numel, shape=x.shape,
            ndim=x.ndim, stride=x.stride, contig=True, metadata={"created_by": "Exp", "ops": "UnaryOps"},
        )
        out = Tensor(
            Dispatcher.dispatch(x=x, ops=UnaryOps.EXP),
            device=x.device, dtype=x.dtype, properties=tp,
        )
        
        if get_graph():
            graph.add_edge(child=out, parents=(x,))
        
        self.x = x
        out._ctx = self
        self.parents = (x,)

        return out
    
    def backward(self, grad_output):
        self.x.grad = grad_output * self.x

class CrossEntropy(Op):
    def forward(self, logits: Tensor, targets: Tensor):
        # NLL(log(softmax(logits)), targets)

        # why not log_softmax here ?
        t1 = Tensor.softmax(logits)
        t2 = -t1.log()[targets]
    
        # if get_graph():
        #     graph.add_edge(child=out, parents=(x,))
        out = Tensor.sum(t2)

        self.x = logits
        self.t1 = t1
        self.targets = targets
        out._ctx = self
        self.parents = (logits,)

        return out
    
    def backward(self, grad_output):
        tp = TensorProperties(
            view=False, offset=0, numel=self.t1.numel, shape=self.t1.shape,
            ndim=self.t1.ndim, stride=self.t1.stride, contig=True
        )
        self.x.grad = Tensor(
            Dispatcher.dispatch(x=self.t1, y=self.targets, ops=CustomOps.CE_BACKWARD),
            device=self.t1.device, dtype=self.t1.dtype, properties=tp
        )
