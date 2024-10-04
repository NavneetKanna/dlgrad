import os
import sys

sys.path.append(os.getcwd())
import unittest

import numpy as np
import torch

from dlgrad import nn
from dlgrad.tensor import Tensor


class TestNN(unittest.TestCase):
    def setUp(self):
        BS, inp_dim, out_dim = 4, 3, 5 
        self.x = Tensor.rand((BS, inp_dim))
        self.fc = nn.Linear(inp_dim, out_dim)
        self.to_fc = torch.nn.Linear(inp_dim, out_dim)

    def test_linear(self):
        dl_out = self.fc(self.x)

        w = self.fc.weight
        b = self.fc.bias
        with torch.no_grad():
            x = torch.tensor(self.x.numpy(), dtype=torch.float32, device="cpu")
            self.to_fc.weight[:] = torch.tensor(w.numpy(), dtype=torch.float32, device="cpu")
            self.to_fc.bias[:] = torch.tensor(b.numpy(), dtype=torch.float32, device="cpu")
            tr_out = self.to_fc(x)

        np.testing.assert_allclose(dl_out.numpy(), tr_out.detach().numpy(), atol=1e-4) 

    def test_linear_with_relu(self):
        dl_out = Tensor.relu(self.fc(self.x))

        w = self.fc.weight
        b = self.fc.bias
        with torch.no_grad():
            x = torch.tensor(self.x.numpy(), dtype=torch.float32, device="cpu")
            self.to_fc.weight[:] = torch.tensor(w.numpy(), dtype=torch.float32, device="cpu")
            self.to_fc.bias[:] = torch.tensor(b.numpy(), dtype=torch.float32, device="cpu")
            tr_out = torch.relu(self.to_fc(x))

        np.testing.assert_allclose(dl_out.numpy(), tr_out.detach().numpy(), atol=1e-4) 


if __name__ == "__main__":
    unittest.main()