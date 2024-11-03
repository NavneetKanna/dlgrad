import unittest
from dlgrad import Tensor
import torch
import numpy as np


class TestOps(unittest.TestCase):
    def test_add(self):
        sh1 = (2, 3)
        sh2 = (2, 3)
        da = Tensor.rand(sh1)
        db = Tensor.rand(sh2)

        dc = da+db

        ta = torch.tensor(da.numpy())
        tb = torch.tensor(db.numpy())

        tc = ta+tb

        np.testing.assert_allclose(tc.numpy(), dc.numpy(), atol=1e-6, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()