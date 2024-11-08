import math
from dlgrad import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        bound = 1 / math.sqrt(in_features)
        self.weight = Tensor.uniform((out_features, in_features), low=-bound, high=bound)
        