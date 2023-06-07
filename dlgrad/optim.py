from typing import TYPE_CHECKING
import numpy as np
from .helper import backward_list

if TYPE_CHECKING:
    from .tensor import Tensor

# It is basically mini-batch GD 
class SGD:
    def __init__(self, obj: object, lr: float) -> None:
        self.lr = lr
        self.parameters: list[Tensor] = []
        # print(obj.__dict__)
        for var in obj.__dict__:
            if 'pool' in var:
                continue
            # if not obj.__dict__[var].weight:
            #     continue
            self.parameters.append(obj.__dict__[var].weight)
            # bias is not supported for conv as of now
            if "conv" in var:
                continue 
            if obj.__dict__[var].bias: self.parameters.append(obj.__dict__[var].bias) 

    def step(self):
        # Update the weights and biases
        for parameters in self.parameters:
            parameters.tensor = parameters.tensor - (self.lr*parameters.grad)

    def zero_grad(self):
        for parameters in backward_list:
            parameters.grad = 0 
        backward_list.clear()
