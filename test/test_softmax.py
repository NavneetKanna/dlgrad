import unittest
from torch import nn
import torch
from dlgrad.tensor import Tensor
from dlgrad.afu import softmax
import numpy as np

class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        m = nn.Softmax(dim=1)
        input = torch.randn(2, 3)
        torch_output = m(input) 

        my_output = softmax(Tensor(input.numpy()))

        np.testing.assert_allclose(torch_output.numpy(), my_output, rtol=0, atol=10**(-2))