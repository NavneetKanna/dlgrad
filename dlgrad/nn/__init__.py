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
