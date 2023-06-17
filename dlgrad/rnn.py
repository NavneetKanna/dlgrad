import math
from typing import Optional
from dlgrad.tensor import Tensor
import numpy as np


# https://towardsdatascience.com/all-you-need-to-know-about-rnns-e514f0b00c7c
# https://medium.com/geekculture/a-look-under-the-hood-of-pytorchs-recurrent-neural-network-module-47c34e61a02d

class RNN():
    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        stdv = 1.0 / math.sqrt(self.hidden_size) 
        
        self.weight_ih: Tensor = Tensor.uniform((self.hidden_size, self.input_size), -stdv, stdv)
        self.weight_hh: Tensor = Tensor.uniform((self.hidden_size, self.hidden_size), -stdv, stdv)

    def __call__(self, input: Tensor, hx: Optional[Tensor]=None):
        assert input.tensor.ndim == 3, "Input dim should be 3, as of now dlgrad supports only batched RNN"
        if hx is not None: self.hx = np.zeros((1*1, input.shape[0], self.hidden_size)) 
        else: self.hx = hx
        
        ih_out = np.zeros(input.shape[0], input.shape[1], 1*self.hidden_size)
        for bs in range(input.shape[0]):
            for tn in range(input.shape[1]):
                out = input[bs, tn] @ self.weight_ih.T
                hx[0, bs] = hx[0, bs] @ self.weight_hh.T
                ht = np.tanh(out + hx[0, bs])
                hx[0, bs] = ht
                ih_out[bs, tn] = ht
        
        return ih_out, self.hx

