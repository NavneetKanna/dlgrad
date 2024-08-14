import math

from dlgrad.tensor import Tensor


class Linear:
    def __init__(self, inp_dim: int, out_dim: int) -> None:
        # TODO: Remove kaiming
        self.weight = Tensor.kaiming_uniform(out_dim, inp_dim)
        bound = 1 / math.sqrt(inp_dim)
        self.bias = Tensor.rand(out_dim, low=-bound, high=bound)

    # x@weight.T + bias
    def __call__(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.weight.shape[-1], f"{x.shape} and {self.weight.shape} does not match"

        return x.linear(self.weight, self.bias)
