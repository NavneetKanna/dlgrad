import os
import sys

sys.path.append(os.getcwd())
import unittest

import numpy as np
import torch

from dlgrad import nn
from dlgrad.tensor import Tensor


class TestNN(unittest.TestCase):
    def test_linear(self):
        BS, inp_dim, out_dim = 4, 3, 5 
        x = Tensor.rand((BS, inp_dim))
        fc = nn.Linear(inp_dim, out_dim)
        dl_out = fc(x)

        w = fc.weight
        b = fc.bias
        with torch.no_grad():
            x = torch.tensor(x.numpy(), dtype=torch.float32, device="cpu")
            fc = torch.nn.Linear(inp_dim, out_dim)
            fc.weight[:] = torch.tensor(w.numpy(), dtype=torch.float32, device="cpu")
            fc.bias[:] = torch.tensor(b.numpy(), dtype=torch.float32, device="cpu")
            tr_out = fc(x)

        np.testing.assert_allclose(dl_out.numpy(), tr_out.detach().numpy(), atol=1e-4) 

    def test_linear_with_relu(self):
        pass


if __name__ == "__main__":
    unittest.main()