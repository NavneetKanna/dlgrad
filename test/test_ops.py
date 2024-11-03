import unittest
from dlgrad import Tensor
import torch
import numpy as np


# TODO: Not an efficient way, need dlgrad to support creating Tensor from np, see tinygrad
def run(shapes: list[tuple], func):
    dla = Tensor.rand(shapes[0])
    dlb = Tensor.rand(shapes[1])
    toa = torch.tensor(dla.numpy())
    tob = torch.tensor(dlb.numpy())

    np.testing.assert_allclose(func(dla, dlb).numpy(), func(toa, tob).numpy(), atol=1e-6, rtol=1e-3)
    

class TestOps(unittest.TestCase):
    def test_add_same_shape(self):
        sh1 = (2, 3)
        sh2 = (2, 3)
        run([sh1, sh2], lambda x, y: x+y)
        sh1 = (100, 100)
        sh2 = (100, 100)
        run([sh1, sh2], lambda x, y: x+y)
        sh1 = (78, 91)
        sh2 = (78, 91)
        run([sh1, sh2], lambda x, y: x+y)


if __name__ == "__main__":
    unittest.main()