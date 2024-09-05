import os
import sys

sys.path.append(os.getcwd())
import unittest

import numpy as np
import torch

from dlgrad.tensor import Tensor


# TODO: Should i use only one framework to be consistent ?
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

    def test_sub(self):
        y = Tensor.rand((3, 2))
        dl_out = self.x-y
        np_out = self.x.numpy() - y.numpy()
        np.testing.assert_equal(dl_out.numpy(), np_out)
        y = Tensor.rand((2))
        dl_out = self.x-y
        np_out = self.x.numpy() - y.numpy()
        np.testing.assert_equal(dl_out.numpy(), np_out)
        y = Tensor.rand((2))
        dl_out = self.x-y
        np_out = self.x.numpy() - y.numpy()
        np.testing.assert_equal(dl_out.numpy(), np_out)

    def test_div(self):
        y = Tensor.rand((3, 2))
        dl_out = self.x/y
        np_out = self.x.numpy() / y.numpy()
        np.testing.assert_equal(dl_out.numpy(), np_out)
        y = Tensor.rand((2))
        dl_out = self.x/y
        np_out = self.x.numpy() / y.numpy()
        np.testing.assert_equal(dl_out.numpy(), np_out)
        y = Tensor.rand((2))
        dl_out = self.x/y
        np_out = self.x.numpy() / y.numpy()
        np.testing.assert_equal(dl_out.numpy(), np_out)

    def test_transpose(self):
        y = Tensor.transpose(self.x)
        np.testing.assert_equal(y.numpy(), self.x.numpy().T)

    def test_sum(self):
        dl_out = self.x.sum()
        np.testing.assert_equal(dl_out.numpy(), self.x.numpy().sum())
        dl_out = self.x.sum(axis=1)
        np.testing.assert_equal(dl_out.numpy(), self.x.numpy().sum(axis=1))
        dl_out = self.x.sum(axis=0)
        np.testing.assert_equal(dl_out.numpy(), self.x.numpy().sum(axis=0))
        dl_out = self.x.sum(keepdim=True)
        np.testing.assert_equal(dl_out.numpy(), self.x.numpy().sum(keepdims=True))
        dl_out = self.x.sum(axis=1, keepdim=True)
        np.testing.assert_equal(dl_out.numpy(), self.x.numpy().sum(axis=1, keepdims=True))
        dl_out = self.x.sum(axis=0, keepdim=True)
        np.testing.assert_equal(dl_out.numpy(), self.x.numpy().sum(axis=0, keepdims=True))

    def test_relu(self):
        dl_out = Tensor.relu(self.x)
        np.testing.assert_equal(dl_out.numpy(), np.maximum(0, self.x.numpy()))
    
    def test_exp(self):
        dl_out = Tensor.exp(self.x)
        np.testing.assert_allclose(dl_out.numpy(), np.exp(self.x.numpy()))

    def test_max(self):
        dl_out = Tensor.max(self.x)
        np.testing.assert_equal(dl_out.numpy(), np.max(self.x.numpy()))

    def test_softmax(self):
        dl_out = Tensor.softmax(self.x, axis=1)
        to_out = torch.softmax(torch.tensor(self.x.numpy()), dim=1)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-7, rtol=0.001)
        dl_out = Tensor.softmax(self.x, axis=0)
        to_out = torch.softmax(torch.tensor(self.x.numpy()), dim=0)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-7, rtol=0.001)

    def test_log(self):
        dl_out = self.x.log()
        to_out = torch.log(torch.tensor(self.x.numpy()))
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-7, rtol=0.001)

    def test_log_softmax(self):
        dl_out = Tensor.log_softmax(self.x)
        to_out = torch.log_softmax(torch.tensor(self.x.numpy()), 1)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-7, rtol=0.001)

    def test_neg(self):
        dl_out = -self.x
        to_out = -torch.tensor(self.x.numpy())
        np.testing.assert_equal(dl_out.numpy(), to_out.numpy())


if __name__ == "__main__":
    unittest.main()