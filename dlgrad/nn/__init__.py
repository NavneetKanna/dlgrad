from dlgrad.tensor import Tensor
import math

class Linear:
    def __init__(self, inp_dim: int, out_dim: int) -> None:
        self.weight = Tensor.kaiming_uniform(out_dim, inp_dim)
        bound = 1 / math.sqrt(inp_dim)
        self.bias = Tensor.rand(out_dim, low=-bound, high=bound)