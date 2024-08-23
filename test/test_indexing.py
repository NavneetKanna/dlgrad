import os
import sys

sys.path.append(os.getcwd())
import unittest

import numpy as np

from dlgrad.tensor import Tensor


class TestNN(unittest.TestCase):
    def test_dim2(self):
        x = Tensor.rand((2, 3))

        np.testing.assert_equal(x[0].numpy(), x.numpy()[0])
        np.testing.assert_equal(x[1].numpy(), x.numpy()[1])
        np.testing.assert_equal(x[0][2].numpy(), x.numpy()[0][2])
        np.testing.assert_equal(x[1][1].numpy(), x.numpy()[1][1])

    def test_dim3(self):
        x = Tensor.rand((2, 3, 3))

        np.testing.assert_equal(x[0].numpy(), x.numpy()[0])
        np.testing.assert_equal(x[1].numpy(), x.numpy()[1])
        np.testing.assert_equal(x[1][0].numpy(), x.numpy()[1][0])
        np.testing.assert_equal(x[1][1].numpy(), x.numpy()[1][1])
        np.testing.assert_equal(x[0][2].numpy(), x.numpy()[0][2])
        np.testing.assert_equal(x[0][2][2].numpy(), x.numpy()[0][2][2])
        np.testing.assert_equal(x[1][0][1].numpy(), x.numpy()[1][0][1])
        np.testing.assert_equal(x[1][1][1].numpy(), x.numpy()[1][1][1])


if __name__ == "__main__":
    unittest.main()