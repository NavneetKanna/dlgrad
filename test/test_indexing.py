import numpy as np
import pytest
import torch
from dlgrad import Tensor


np_data = np.random.uniform(size=(10, 20)).astype(np.float32)

a = Tensor(np_data)
ta = torch.tensor(np_data)

b = a[2:4]
tb = ta[2:4]
def test():
    np.testing.assert_allclose(b.numpy(), tb.numpy(), atol=1e-6, rtol=1e-3)

b = a[1:1]
tb = ta[1:1]
def test():
    np.testing.assert_allclose(b.numpy(), tb.numpy(), atol=1e-6, rtol=1e-3)

b = a[10:1]
tb = ta[10:1]
def test():
    np.testing.assert_allclose(b.numpy(), tb.numpy(), atol=1e-6, rtol=1e-3)

b = a[1:2]
tb = ta[1:2]
def run(func):
    np.testing.assert_allclose(func(a, b).numpy(), func(ta, tb).numpy(), atol=1e-6, rtol=1e-3)

run(lambda x, y: x+y)
run(lambda x, y: x-y)
run(lambda x, y: x*y)
run(lambda x, y: x/y)
