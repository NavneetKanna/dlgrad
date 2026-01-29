import math

from dlgrad import Tensor
from dlgrad.device import Device
from dlgrad.nn import datasets, optim, utils  # noqa: F401


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Device = Device.CPU) -> None:
        bound = 1 / math.sqrt(in_features)
        self.weight = Tensor.uniform((out_features, in_features), low=-bound, high=bound, requires_grad=True, device=device)
        self.bias = Tensor.uniform((1, out_features), low=-bound, high=bound, requires_grad=True, device=device) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        return x.linear(self.weight, self.bias)

class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor.uniform((self.num_embeddings, self.embedding_dim), requires_grad=True)

    def __call__(self, idx: Tensor) -> Tensor:
        # if not idx.dtype == DType.int:
        #     raise TypeError(f"Expected integer indicies, but got {idx.dtype}")
        return Tensor.embedding(self.weight, idx)

class RMSNorm:
    def __init__(self, dim: int, eps: int = 1e-6, elementwise_affine: bool = True) -> None:
        self.eps = eps
        self.weight = Tensor.ones_like(Tensor.rand((1, dim)), requires_grad=True) if elementwise_affine else None

    def _norm(self, x: Tensor) -> Tensor:
        t = x**2
        t = t.mean(t.ndim - 1, keepdim=True)
        t = (t + self.eps).rsqrt()
        return x * t

    def __call__(self, x: Tensor) -> Tensor:
        x = self._norm(x)
        return x if self.weight is None else x * self.weight


