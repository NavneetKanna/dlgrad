import math

from dlgrad import Tensor
from dlgrad.nn import datasets, optim, utils  # noqa: F401


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        bound = 1 / math.sqrt(in_features)
        self.weight = Tensor.uniform((out_features, in_features), low=-bound, high=bound, requires_grad=True)
        self.bias = Tensor.uniform((1, out_features), low=-bound, high=bound, requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        # print(self.weight.numpy())
        # print("---")
        return x.linear(self.weight, self.bias)
