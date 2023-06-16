import unittest
from torch import nn
import torch
from dlgrad.tensor import Tensor
from dlgrad.loss import crossentropy
import numpy as np

class TestCrossEntropy(unittest.TestCase):
    def test_cross_entropy(self):
        torch_loss = nn.CrossEntropyLoss()

        input = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(5)
        my_inp = Tensor(input.detach().numpy())
        my_target = Tensor(target.detach().numpy())

        torch_output = torch_loss(input, target)
        # torch_output.backward()

        my_output = crossentropy(my_inp, my_target)

        self.assertEqual(torch_output.detach().numpy().astype(np.float32).round(2), my_output.tensor.astype(np.float32).round(2))



        # Example of target with class probabilities
        # input2 = torch.randn(3, 5, requires_grad=True)
        # target2 = torch.randn(3, 5).softmax(dim=1)
        # my_inp2 = Tensor(input2.detach().numpy())
        # my_target2 = Tensor(target2.detach().numpy())

        # torch_output2 = torch_loss(input2, target2)
        # # output.backward()
        # my_output2 = crossentropy(my_inp2, my_target2)

        # self.assertEqual(torch_output2.detach().numpy().astype(np.float32).round(2), my_output2.tensor.astype(np.float32).round(2))
