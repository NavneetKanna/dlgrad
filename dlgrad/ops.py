from dlgrad.buffer import Buffer
from dlgrad.dtype import Scalar
from dlgrad.helpers import CustomOps, broadcast_shapes, check_broadcast
from dlgrad.tensor import OP


# ------------ Unary Ops -----------
class Pow(OP):
    def forward(self, x: Buffer, y: float) -> Buffer:
        self.x = x
        self.y = y  # We assume y is a scalar float/int for RMSNorm (x**2)
        return x**y

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        # Equation: grad * p * x^(p-1)

        # 1. Calculate x^(p-1)
        # Note: We rely on the buffer's pow method
        base_derivative = self.x ** (self.y - 1.0)

        # 2. Multiply by p
        scale = base_derivative * self.y

        # 3. Multiply by upstream gradient
        return (upstream_grad * scale,)

class Reshape(OP):
    def forward(self, x: Buffer, shape: tuple) -> Buffer:
        self.input_shape = x.shape
        return x.reshape(shape)

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        return (upstream_grad.reshape(self.input_shape),)

class Permute(OP):
    def forward(self, x: Buffer, order: tuple) -> Buffer:
        self.order = order
        return x.permute(order)

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        # The gradient of a permute is the inverse permute.

        inv_order = [0] * len(self.order)
        for i, p in enumerate(self.order):
            inv_order[p] = i

        return (upstream_grad.permute(tuple(inv_order)),)

class Transpose(OP):
    def forward(self, x: Buffer, dim0: int, dim1: int) -> Buffer:
        self.dim0 = dim0
        self.dim1 = dim1
        return x.transpose(dim0, dim1)

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        return (upstream_grad.transpose(self.dim1, self.dim0),)

class Sum(OP):
    def forward(self, x: Buffer, dim: int = -1, keepdim: bool = False) -> Buffer:
        self.inp_shape = x.shape
        self.device = x.device
        self.dtype = x.dtype
        self.keepdim = keepdim
        self.dim = dim
        return x.sum(dim=dim, keepdim=keepdim)

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        t = Buffer.full(shape=self.inp_shape, fill_value=1.0, device=self.device, dtype=self.dtype)
        if not self.keepdim:
            upstream_grad.unsqueeze(self.dim)
        return (t*upstream_grad,)

class Mean(OP):
    def forward(self, x: Buffer, dim: int = -1, keepdim: bool = False) -> Buffer:
        self.inp_shape = x.shape
        self.device = x.device
        self.dtype = x.dtype
        self.keepdim = keepdim
        self.dim = dim
        if dim == -1:  # reduce over ALL elements
            self.N = x.numel
        else:
            self.N = x.shape[dim]
        return x.mean(dim=dim, keepdim=keepdim)

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        scale = 1.0 / self.N
        grad = Buffer.full(self.inp_shape, fill_value=scale, device=self.device, dtype=self.dtype)
        if not self.keepdim:
            upstream_grad.unsqueeze(self.dim)
        return (grad * upstream_grad,)

class Max(OP):
    def forward(self, x: Buffer, dim: int = -1, keepdim: bool = False) -> Buffer:
        self.inp_shape = x.shape
        self.device = x.device
        self.dim = dim
        self.x = x
        self.keepdim = keepdim
        self.out = x.max(dim=dim, keepdim=keepdim)
        return self.out

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        if not self.keepdim:
            self.out.unsqueeze(self.dim)

        max_with_1s = self.x == self.out

        if not self.keepdim:
            upstream_grad.unsqueeze(self.dim)

        return (max_with_1s*upstream_grad,)

class Tril(OP):
    def forward(self, x: Buffer, k: int = 0) -> Buffer:
        # NOTE: Currently supports only 2D
        H, W = x.shape[0], x.shape[1]
        rows = Buffer.arange((H, 1), x.device)
        cols = Buffer.arange((1, W), x.device)

        mask = cols <= (rows + k)

        self.mask = mask

        return mask.where(x, 0.0)

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        return (self.mask.where(upstream_grad, 0.0),)


class Exp(OP):
    def forward(self, x: Buffer) -> Buffer:
        self.out = x.exp()
        return self.out

    def backward(self, upstream_grad: Buffer) -> Buffer:
        return (upstream_grad * self.out,)

class Log(OP):
    def forward(self, x: Buffer) -> Buffer:
        self.x = x
        return x.log()

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        return (upstream_grad / self.x,)

class Relu(OP):
    def forward(self, x: Buffer) -> Buffer:
        self.out = x.relu()
        return self.out

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        return ((self.out.where(inp=1.0, other=0.0)) * upstream_grad,)

class Sigmoid(OP):
    def forward(self, x: Buffer) -> Buffer:
        self.out = x.sigmoid()
        return self.out

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        return ((self.out * -(self.out - 1.0)) * upstream_grad,)

class LeakyRelu(OP):
    def forward(self, x: Buffer, neg_slope: Scalar = 0.01) -> Buffer:
        self.neg_slope = neg_slope
        self.out = x.leaky_relu(neg_slope=neg_slope)
        return self.out

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        return ((self.out.where(inp=1.0, other=self.neg_slope)) * upstream_grad,)

class Tanh(OP):
    def forward(self, x: Buffer) -> Buffer:
        self.out = x.tanh()
        return self.out

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        # TODO: Remove buffer.full() and add support for scalar in buffer sub
        return ((Buffer.full(self.out.shape, 1.0, self.out.device, self.out.dtype) - self.out**2) * upstream_grad,)

class Sqrt(OP):
    def forward(self, x: Buffer) -> Buffer:
        self.out = x.sqrt()
        return self.out

    def backward(self, grad_output: Buffer) -> tuple[Buffer]:
        return (grad_output / (self.out*2),)

class RSqrt(OP):
    def forward(self, x: Buffer) -> Buffer:
        self.out = x.rsqrt()
        return self.out

    def backward(self, grad_output: Buffer) -> tuple[Buffer]:
        return (grad_output * (self.out**3 * -0.5),)

class Clamp(OP):
    def forward(self, x: Buffer, min: int | None, max: int | None) -> Buffer:
        self.min = min
        self.max = max
        self.out = x.clamp(min, max)
        return self.out

    def backward(self, grad_output: Buffer) -> tuple[Buffer]:
        return (grad_output.clamp(self.min, self.max))

class Squeeze(OP):
    def forward(self, x: Buffer, dim: list[int] | int) -> Buffer:
        self.dim = dim
        x.squeeze(dim)
        return x

    def backward(self, grad_output: Buffer) -> tuple[Buffer]:
        grad_output.unsqueeze(self.dim)
        return (grad_output,)

class Unsqueeze(OP):
    def forward(self, x: Buffer, dim: list[int] | int) -> Buffer:
        self.dim = dim
        x.unsqueeze(dim)
        return x

    def backward(self, grad_output: Buffer) -> tuple[Buffer]:
        grad_output.squeeze(self.dim)
        return (grad_output,)

# ------------ Binary Ops -----------

class Add(OP):
    def forward(self, x: Buffer, y: Buffer) -> Buffer:
        self.x = x
        self.y = y
        if check_broadcast(x.shape, y.shape):
            return x + y

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
        grad_x = self.reduce_grad_for_broadcasting(upstream_grad, self.x.shape) if self.req_grad[0] else None
        grad_y = self.reduce_grad_for_broadcasting(upstream_grad, self.y.shape) if self.req_grad[1] else None
        return grad_x, grad_y


class Sub(OP):
    def forward(self, x: Buffer, y: Buffer) -> Buffer:
        self.x = x
        self.y = y
        if check_broadcast(x.shape, y.shape):
            return x - y

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
        grad_x = self.reduce_grad_for_broadcasting(upstream_grad, self.x.shape) if self.req_grad[0] else None
        grad_y = self.reduce_grad_for_broadcasting(-upstream_grad, self.y.shape) if self.req_grad[1] else None
        return grad_x, grad_y


class Mul(OP):
    def forward(self, x: Buffer, y: Buffer) -> Buffer:
        self.x = x
        self.y = y
        if check_broadcast(x.shape, y.shape):
            return x*y

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
        grad_x = self.reduce_grad_for_broadcasting(upstream_grad*self.y, self.x.shape) if self.req_grad[0] else None
        grad_y = self.reduce_grad_for_broadcasting(upstream_grad*self.x, self.y.shape) if self.req_grad[1] else None
        return grad_x, grad_y


class Div(OP):
    def forward(self, x: Buffer, y: Buffer | Scalar) -> Buffer:
        self.x = x
        self.y = y
        if check_broadcast(x.shape, y.shape):
            return x/y

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer | None, Buffer | None]:
        grad_x = self.reduce_grad_for_broadcasting(upstream_grad/self.y, self.x.shape) if self.req_grad[0] else None
        grad_y = self.reduce_grad_for_broadcasting((-upstream_grad*self.x)/self.y**2, self.y.shape) if self.req_grad[1] else None
        return grad_x, grad_y

# def unbroadcast(grad: Buffer, original_shape: tuple) -> Buffer:
#     for i, (grad_dim, in_dim) in enumerate(zip(grad.shape, original_shape)):
#         if in_dim == 1 and grad_dim > 1:
#             grad = grad.sum(i, keepdim=True)

#     return grad

def unbroadcast(grad: Buffer, original_shape: tuple) -> Buffer:
    while grad.ndim > len(original_shape):
        grad = grad.sum(0, keepdim=False)

    for i, (grad_dim, in_dim) in enumerate(zip(grad.shape, original_shape)):
        if in_dim == 1 and grad_dim > 1:
            grad = grad.sum(i, keepdim=True)

    return grad

class MatMul(OP):
    def forward(self, x: Buffer, y: Buffer) -> Buffer:
        # We need to save the shapes used for the internal matmul
        # (the ones produced by broadcast_shapes)
        self.p1, self.p2, self.out_shape = broadcast_shapes(x.shape, y.shape)
        self.x_old_shape = x.shape
        self.y_old_shape = y.shape
        self.x = x
        self.y = y
        return x @ y

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        # 1. Fix: Catch the return value of reshape!
        # If upstream_grad has prepended dims that were squeezed out or diff shape, align it.
        if upstream_grad.shape != self.out_shape:
            upstream_grad = upstream_grad.reshape(self.out_shape)

        # 2. Fix: Use .reshape() method instead of manual Buffer()
        # This uses your new logic to create a proper View with correct metadata/strides.
        # This handles the case where shapes differ only by singleton dimensions (e.g. (3,4) -> (1,3,4))
        x_reshaped = self.x.reshape(self.p1)
        y_reshaped = self.y.reshape(self.p2)

        # 3. Compute transposes (Standard Backprop)
        # Gradient wrt A = Grad @ B.T
        # Gradient wrt B = A.T @ Grad
        # We use ndim-2 and ndim-1 to always swap the last two matrix dimensions
        t1 = x_reshaped.transpose(x_reshaped.ndim - 2, x_reshaped.ndim - 1)
        t2 = y_reshaped.transpose(y_reshaped.ndim - 2, y_reshaped.ndim - 1)

        # 4. Compute gradients via MatMul
        grad_x = upstream_grad @ t2
        grad_y = t1 @ upstream_grad

        # 5. Unbroadcast back to original shapes (Sum out the broadcasted dims)
        grad_x = unbroadcast(grad_x, self.x_old_shape)
        grad_y = unbroadcast(grad_y, self.y_old_shape)

        return (grad_x, grad_y)

class CrossEntropy(OP):
    def forward(self, logits: Buffer, target: Buffer, dim: int = 1) -> Buffer:
        assert logits.shape[0] == target.shape[0]
        self.target = target
        self.dim = dim

        # Stable log_softmax
        max_val = logits.max(dim=dim, keepdim=True)
        shifted = logits - max_val
        exp_vals = shifted.exp()
        sum_exp = exp_vals.sum(dim=dim, keepdim=True)
        self.log_softmax = shifted - sum_exp.log()  # Save full (N, V) log_softmax

        # Gather log probs for targets: (N,)
        tmp = Buffer.ce_forward(self.log_softmax, self.target)

        # N = number of samples/tokens (B or B*T after flattening in model)
        self.N = float(logits.shape[0])

        # Mean reduction
        return -tmp.sum()

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        # Recompute softmax probs (N, V)
        tmp = self.log_softmax.exp()

        # In-place: tmp -= one_hot(target) => tmp = p - one_hot
        Buffer.ce_backward(op=CustomOps.CE_BACKWARD, device=tmp.device, x=tmp, target=self.target)

        # Scale for mean reduction and chain rule
        grad_out = tmp * upstream_grad  # upstream_grad usually 1.0 (scalar), broadcasts
        grad_out = grad_out / float(self.N)

        return (grad_out,)

class CCrossEntropy(OP):
    def forward(self, logits: Buffer, target: Buffer, dim: int = 1) -> Buffer:
        assert logits.shape[0] == target.shape[0], f"logits shape[0] and target shape[0] does not match {logits.shape} != {target.shape}"  # noqa: E501

        self.target = target
        t = logits.max(dim=dim, keepdim=True)
        m = logits - t
        e = m.exp()
        ss = e.sum(dim=dim, keepdim=True)
        self.log_softmax_output = m - ss.log()
        tmp = Buffer.ce_forward(self.log_softmax_output, self.target)

        self.numel = tmp.shape[0]

        # self.div_factor = tmp.numel
        # return -tmp.sum() / float(self.div_factor)
        return -tmp.mean()
        return -tmp.sum()

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        tmp = self.log_softmax_output.exp()
        Buffer.ce_backward(op=CustomOps.CE_BACKWARD, device=tmp.device, x=tmp, target=self.target)
        grad_out = (tmp * upstream_grad) / float(self.numel)
        return (grad_out,)
        return (tmp,)

# https://publish.obsidian.md/kamilelukosiute/pytorch/What+does+BCEWithLogits+actually+do%3F
class BCEWithLogitsLoss(OP):
    def forward(self, probs: Buffer, target: Buffer) -> Buffer:
        assert probs.shape == target.shape, f"shape mismatch {probs.shape} vs {target.shape}"

        max_val = (-probs).clamp(min=0) # or (-probs).relu()
        self.probs = probs
        self.target = target
        self.loss = ((1.0-target) * probs) + max_val + ((-max_val).exp() + (-(probs+max_val)).exp()).log()
        return self.loss.mean()

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        N = self.probs.numel
        grad = ((self.probs.sigmoid() - self.target) * upstream_grad) / float(N)
        return (grad,)

class Embedding(OP):
    def forward(self, weight: Buffer, idx: Buffer) -> Buffer:
        self.idx = idx
        self.weight = weight
        return weight.embedding(idx)

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        return (self.weight.embedding(self.idx, backward=True, upstream_grad=upstream_grad),)

class Where(OP):
    def forward(self, cond: Buffer, inp: Buffer | Scalar, other: Buffer | Scalar) -> Buffer:
        self.cond = cond
        return cond.where(inp, other)

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        grad_inp = self.cond.where(upstream_grad, 0.0)
        grad_other = self.cond.where(0.0, upstream_grad)
        return (None, grad_inp, grad_other)

class MaskedFill(OP):
    def forward(self, data: Buffer, mask: Buffer, value: Scalar) -> Buffer:
        self.mask = mask
        return Buffer.masked_fill(x=data, mask=mask, val=value)

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        grad_input = Buffer.masked_fill(x=upstream_grad, mask=self.mask, val=0.0)
        return (grad_input, None, None)

class Dropout(OP):
    def forward(self, x: Buffer, p: Scalar) -> Buffer:
        self.p = p

        if p == 1.0:
            return Buffer.full(x.shape, 0.0, x.device, x.dtype)
        if p == 0.0:
            return x

        self.mask = Buffer.uniform(x.shape, x.device, x.dtype, low=0.0, high=1.0) >= p
        self.scale = 1.0 / (1.0 - p)
        out = self.mask.where(x, 0.0)

        return out * self.scale

    def backward(self, upstream_grad: Buffer) -> tuple[Buffer]:
        if self.p == 1.0:
            return (Buffer.full(upstream_grad.shape, 0.0, upstream_grad.device, upstream_grad.dtype),)
        if self.p == 0.0:
            return (upstream_grad,)
        return (self.mask.where(upstream_grad, 0.0) * self.scale)

