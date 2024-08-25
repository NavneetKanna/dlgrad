import os
import sys

sys.path.append(os.getcwd())
import unittest

import numpy as np

from dlgrad.tensor import Tensor


class TestOps(unittest.TestCase):
    def setUp(self):
        self.x = Tensor.rand((3, 2))

    def test_add(self):
        y = Tensor.rand((3, 2))
        dl_out = self.x+y
        np_out = self.x.numpy() + y.numpy()
        np.testing.assert_equal(dl_out.numpy(), np_out)
        y = Tensor.rand((2))
        dl_out = self.x+y
        np_out = self.x.numpy() + y.numpy()
        np.testing.assert_equal(dl_out.numpy(), np_out)
        y = Tensor.rand((2))
        dl_out = self.x+y
        np_out = self.x.numpy() + y.numpy()
        np.testing.assert_equal(dl_out.numpy(), np_out)

    # def test_div(self):
    #     y = Tensor.rand((3, 2))
    #     dl_out = self.x/y
    #     np_out = self.x.numpy() / y.numpy()
    #     np.testing.assert_equal(dl_out.numpy(), np_out)
    #     y = Tensor.rand((2))
    #     dl_out = self.x/y
    #     np_out = self.x.numpy() / y.numpy()
    #     np.testing.assert_equal(dl_out.numpy(), np_out)
    #     y = Tensor.rand((2))
    #     dl_out = self.x/y
    #     np_out = self.x.numpy() / y.numpy()
    #     np.testing.assert_equal(dl_out.numpy(), np_out)

    def test_transpose(self):
        y = Tensor.transpose(self.x)
        np.testing.assert_equal(y.numpy(), self.x.numpy().T)

    def test_sum(self):
        dl_out = self.x.sum()
        np.testing.assert_equal(dl_out.numpy(), self.x.numpy().sum())
        
    def test_relu(self):
        dl_out = Tensor.relu(self.x)
        np.testing.assert_equal(dl_out.numpy(), np.maximum(0, self.x.numpy()))
    
    def test_exp(self):
        dl_out = Tensor.exp(self.x)
        np.testing.assert_equal(dl_out.numpy(), np.exp(self.x.numpy()))

if __name__ == "__main__":
    unittest.main()