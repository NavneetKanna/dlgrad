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

@pytest.mark.parametrize("shapes", [
    [(2, 3, 4), (1, 3, 4)],
    [(2, 3), (1, 3)],
    [(2, 3), (2, 3)],
])
def test_div_backward(shapes):
    run(shapes, lambda x, y: x/y)

@pytest.mark.parametrize("shapes", [
    [(2, 3)],
    [(10, 10)]
])
def test_relu_backward(shapes):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)

        dlgrad_data = Tensor(np_data, requires_grad=True)
        torch_data = torch.tensor(np_data, requires_grad=True)

        dlgrad_data.relu().sum().backward()
        torch_data.relu().sum().backward()

        np.testing.assert_allclose(dlgrad_data.grad.numpy(), torch_data.grad.numpy(), atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("shapes", [
    [(4, 3)],
    [(20, 40)],
])
def test_max_2d_backward(shapes):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)

        dlgrad_data = Tensor(np_data, requires_grad=True)
        torch_data = torch.tensor(np_data, requires_grad=True)

        dlgrad_data.max(dim=0).sum().backward()
        to_out, _ = torch_data.max(dim=0)
        to_out.sum().backward()
        np.testing.assert_allclose(dlgrad_data.grad.numpy(), torch_data.grad.numpy(), atol=1e-6, rtol=1e-3)

        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, requires_grad=True)
        torch_data = torch.tensor(np_data, requires_grad=True)

        dlgrad_data.max(dim=1).sum().backward()
        to_out, _ = torch_data.max(dim=1)
        to_out.sum().backward()
        np.testing.assert_allclose(dlgrad_data.grad.numpy(), torch_data.grad.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(2, 3)]
])
def test_ce_backward(shapes):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        np_target = np.array([[1, 2]]).astype(np.float32).reshape(-1, 1)

        dlgrad_data = Tensor(np_data, requires_grad=True)
        dlgrad_target = Tensor(np_target)
        torch_data = torch.tensor(np_data, requires_grad=True)
        torch_target = torch.tensor(np_target, dtype=torch.long).squeeze()

        dlgrad_data.cross_entropy_loss(dlgrad_target).backward()
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
        loss(torch_data, torch_target).backward()

        np.testing.assert_allclose(dlgrad_data.grad.numpy(), torch_data.grad.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(2, 3)]
])
def test_log_sft_backward(shapes):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)

        dlgrad_data = Tensor(np_data, requires_grad=True)
        torch_data = torch.tensor(np_data, requires_grad=True)

        dlgrad_data.log_softmax(dim=1).sum().backward()
        to_out = torch.nn.LogSoftmax(dim=1)
        to_out(torch_data).sum().backward()

        np.testing.assert_allclose(dlgrad_data.grad.numpy(), torch_data.grad.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (3, 2)],
    [(10, 10), (10, 10)]
])
def test_matmul_backward(shapes):
    run(shapes, lambda x, y: x@y)
    