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
        self.weight = Tensor.uniform((self.num_embeddings, self.embedding_dim))

    def __call__(self, idx: Tensor) -> None:
        if not idx.dtype == "int":
            raise TypeError(f"Expected integer indicies, but got {idx.dtype}")


