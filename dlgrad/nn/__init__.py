from dlgrad.tensor import Tensor

class Linear:
    def __init__(self, inp_dim: int, out_dim: int) -> None:
        self.weight = Tensor.kaiming_uniform(out_dim, inp_dim, fan_in=inp_dim)