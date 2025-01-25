import numpy as np
import pytest
import torch

from dlgrad import Tensor


# TODO: Test with one tensor req grad is none
def run(shapes: list[tuple], func):
    np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
    dlgrad_data = [Tensor(data, requires_grad=True) for data in np_data]
    torch_data = [torch.tensor(data, requires_grad=True) for data in np_data]

    func(*dlgrad_data).sum().backward()
    func(*torch_data).sum().backward()

    np.testing.assert_allclose(dlgrad_data[0].grad.numpy(), torch_data[0].grad.numpy(), atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(dlgrad_data[1].grad.numpy(), torch_data[1].grad.numpy(), atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("shapes", [
    [(2, 3, 4), (1, 3, 4)],
    [(2, 3), (1, 3)],
    [(2, 3), (2, 3)],
])
def test_add_backward(shapes):
    run(shapes, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", [
    [(2, 3, 4), (1, 3, 4)],
    [(2, 3), (1, 3)],
    [(2, 3), (2, 3)],
])
def test_sub_backward(shapes):
    run(shapes, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", [
    [(2, 3, 4), (1, 3, 4)],
    [(2, 3), (1, 3)],
    [(2, 3), (2, 3)],
])
def test_mul_backward(shapes):
    run(shapes, lambda x, y: x*y)


