import numpy as np
import pytest
import torch
from dlgrad import Tensor


np_data = np.random.uniform(size=(10, 20)).astype(np.float32)

for d in ["cpu", "metal"]:
    a = Tensor(np_data, device=d)
    if d == "metal":
        d = "mps"
    ta = torch.tensor(np_data, device=d)

    b = a[2:4]
    tb = ta[2:4]
    def test():
        np.testing.assert_allclose(b.numpy(), tb.cpu().numpy(), atol=1e-6, rtol=1e-3)

    b = a[1:1]
    tb = ta[1:1]
    def test():
        np.testing.assert_allclose(b.numpy(), tb.cpu().numpy(), atol=1e-6, rtol=1e-3)

    b = a[10:1]
    tb = ta[10:1]
    def test():
        np.testing.assert_allclose(b.numpy(), tb.cpu().numpy(), atol=1e-6, rtol=1e-3)

    b = a[1:2]
    tb = ta[1:2]
    def run(func):
        np.testing.assert_allclose(func(a, b).numpy(), func(ta, tb).cpu().numpy(), atol=1e-6, rtol=1e-3)

    run(lambda x, y: x+y)
    run(lambda x, y: x-y)
    run(lambda x, y: x*y)
    run(lambda x, y: x/y)
