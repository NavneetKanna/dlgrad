import numpy as np
import pytest
import torch

from dlgrad import Tensor

# TODO: Test NaN's
# TODO: Test Tensor(Scalar)
# TODO: Test ones like
# TODO: Test with shapes containing 1 for sum

# Thanks to tinygrad for the template
def run(shapes: list[tuple], func):
    np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
    dlgrad_data = [Tensor(data) for data in np_data]
    torch_data = [torch.tensor(data) for data in np_data]

    np.testing.assert_allclose(func(*dlgrad_data).numpy(), func(*torch_data).numpy(), atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("shapes", [
    [(2, 3, 4), (1, 3, 4)],
    [(2, 3, 4), (2, 1, 4)],
    [(2, 3, 4), (2, 3, 1)],
    [(2, 3, 4), (1, 1, 4)],
    [(2, 3, 4), (1, 3, 1)],
    [(2, 3, 4), (2, 1, 1)],
    [(2, 3, 4), (1, 1, 1)],
    [(2, 3, 4), (2, 3, 4)],
    [(2, 3), (1, 3)],
    [(2, 3), (2, 1)],
    [(2, 3), (1, 1)],
    [(2, 3), (2, 3)]
])
def test_add(shapes):
    run(shapes, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", [
    [(1, 3, 4), (2, 3, 4)],
    [(2, 1, 4), (2, 3, 4)],
    [(2, 3, 1), (2, 3, 4)],
    [(1, 1, 4), (2, 3, 4)],
    [(1, 3, 1), (2, 3, 4)],
    [(2, 1, 1), (2, 3, 4)],
    [(1, 1, 1), (2, 3, 4)],
    [(2, 3, 4), (2, 3, 4)],
    [(1, 3), (2, 3)],
    [(2, 1), (2, 3)],
    [(1, 1), (2, 3)],
    [(2, 3), (2, 3)]
])
def test_add_diff_shape_reverse(shapes):
    run(shapes, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", [
    [(2, 3, 4), (1, 3, 4)],
    [(2, 3, 4), (2, 1, 4)],
    [(2, 3, 4), (2, 3, 1)],
    [(2, 3, 4), (1, 1, 4)],
    [(2, 3, 4), (1, 3, 1)],
    [(2, 3, 4), (2, 1, 1)],
    [(2, 3, 4), (1, 1, 1)],
    [(2, 3, 4), (2, 3, 4)],
    [(2, 3), (1, 3)],
    [(2, 3), (2, 1)],
    [(2, 3), (1, 1)],
    [(2, 3), (2, 3)]
])
def test_sub(shapes):
    run(shapes, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", [
    [(1, 3, 4), (2, 3, 4)],
    [(2, 1, 4), (2, 3, 4)],
    [(2, 3, 1), (2, 3, 4)],
    [(1, 1, 4), (2, 3, 4)],
    [(1, 3, 1), (2, 3, 4)],
    [(2, 1, 1), (2, 3, 4)],
    [(1, 1, 1), (2, 3, 4)],
    [(2, 3, 4), (2, 3, 4)],
    [(1, 3), (2, 3)],
    [(2, 1), (2, 3)],
    [(1, 1), (2, 3)],
    [(2, 3), (2, 3)]
])
def test_sub_diff_shape_reverse(shapes):
    run(shapes, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", [
    [(2, 3, 4), (1, 3, 4)],
    [(2, 3, 4), (2, 1, 4)],
    [(2, 3, 4), (2, 3, 1)],
    [(2, 3, 4), (1, 1, 4)],
    [(2, 3, 4), (1, 3, 1)],
    [(2, 3, 4), (2, 1, 1)],
    [(2, 3, 4), (1, 1, 1)],
    [(2, 3, 4), (2, 3, 4)],
    [(2, 3), (1, 3)],
    [(2, 3), (2, 1)],
    [(2, 3), (1, 1)],
    [(2, 3), (2, 3)]
])
def test_mul(shapes):
    run(shapes, lambda x, y: x*y)

@pytest.mark.parametrize("shapes", [
    [(1, 3, 4), (2, 3, 4)],
    [(2, 1, 4), (2, 3, 4)],
    [(2, 3, 1), (2, 3, 4)],
    [(1, 1, 4), (2, 3, 4)],
    [(1, 3, 1), (2, 3, 4)],
    [(2, 1, 1), (2, 3, 4)],
    [(1, 1, 1), (2, 3, 4)],
    [(2, 3, 4), (2, 3, 4)],
    [(1, 3), (2, 3)],
    [(2, 1), (2, 3)],
    [(1, 1), (2, 3)],
    [(2, 3), (2, 3)]
])
def test_mul_diff_shape_reverse(shapes):
    run(shapes, lambda x, y: x*y)

@pytest.mark.parametrize("shapes", [
    [(2, 3, 4), (1, 3, 4)],
    [(2, 3, 4), (2, 1, 4)],
    [(2, 3, 4), (2, 3, 1)],
    [(2, 3, 4), (1, 1, 4)],
    [(2, 3, 4), (1, 3, 1)],
    [(2, 3, 4), (2, 1, 1)],
    [(2, 3, 4), (1, 1, 1)],
    [(2, 3, 4), (2, 3, 4)],
    [(2, 3), (1, 3)],
    [(2, 3), (2, 1)],
    [(2, 3), (1, 1)],
    [(2, 3), (2, 3)]
])
def test_div(shapes):
    run(shapes, lambda x, y: x/y)

@pytest.mark.parametrize("shapes", [
    # [(1, 3, 4), (2, 3, 4)],
    # [(2, 1, 4), (2, 3, 4)],
    # [(2, 3, 1), (2, 3, 4)],
    # [(1, 1, 4), (2, 3, 4)],
    # [(1, 3, 1), (2, 3, 4)],
    # [(2, 1, 1), (2, 3, 4)],
    # [(1, 1, 1), (2, 3, 4)],
    # [(2, 3, 4), (2, 3, 4)],
    [(1, 3), (2, 3)],
    # [(2, 1), (2, 3)],
    # [(1, 1), (2, 3)],
    # [(2, 3), (2, 3)]
])
def test_div_diff_shape_reverse(shapes):
    run(shapes, lambda x, y: x/y)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (3, 2)],
    [(100, 100), (100, 100)],
    [(78, 91), (91, 10)],
])
def test_matmul(shapes):
    run(shapes, lambda x, y: x@y)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (2, 3)],
    [(78, 91), (66, 91)],
])
def test_transpose_diff_tensors(shapes):
    run(shapes, lambda x, y: x@y.T)

@pytest.mark.parametrize("shapes", [
    [(66, 91), (66, 78)],
])
def test_transpose_diff_tensors_reverse(shapes):
    run(shapes, lambda x, y: x.T@y)

@pytest.mark.parametrize("shapes", [
    [(2, 3)],
    [(78, 91)],
])
def test_transpose_same_tensors(shapes):
    run(shapes, lambda x: x@x.T)

@pytest.mark.parametrize("shapes", [
    [(4, 3, 2)],
    [(20, 40, 30)],
])
def test_sum_3d(shapes):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data)
        torch_data = torch.tensor(np_data)

        dl_out = dlgrad_data.sum(dim=0)
        to_out = torch_data.sum(dim=0)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.sum(dim=1)
        to_out = torch_data.sum(dim=1)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.sum(dim=2)
        to_out = torch_data.sum(dim=2)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.sum()
        to_out = torch_data.sum()
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(4, 3)],
    [(20, 40)],
])
def test_sum_2d(shapes):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data)
        torch_data = torch.tensor(np_data)

        dl_out = dlgrad_data.sum(dim=0)
        to_out = torch_data.sum(dim=0, keepdim=True)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.sum(dim=1)
        to_out = torch_data.sum(dim=1, keepdim=True)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.sum()
        to_out = torch_data.sum()
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(4, 3, 2)],
    [(20, 40, 30)],
])
def test_max_3d(shapes):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data)
        torch_data = torch.tensor(np_data)

        dl_out = dlgrad_data.max(dim=0)
        to_out, _ = torch_data.max(dim=0)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.max(dim=1)
        to_out, _ = torch_data.max(dim=1)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.max(dim=2)
        to_out, _ = torch_data.max(dim=2)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.max()
        to_out = torch_data.max()
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(4, 3)],
    [(20, 40)],
])
def test_max_2d(shapes):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data)
        torch_data = torch.tensor(np_data)

        dl_out = dlgrad_data.max(dim=0)
        to_out, _ = torch_data.max(dim=0, keepdim=True)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.max(dim=1)
        to_out, _ = torch_data.max(dim=1, keepdim=True)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.max()
        to_out = torch_data.max()
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(2, 3)],
    [(10, 10)]
])
def test_relu(shapes):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data)
        torch_data = torch.tensor(np_data)

        dl_out = dlgrad_data.relu()
        to_out = torch_data.relu()
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(2, 3)],
    [(4, 10, 2)]
])
def test_exp(shapes):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data)
        torch_data = torch.tensor(np_data)

        dl_out = dlgrad_data.exp()
        to_out = torch_data.exp()
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(2, 3)],
    [(4, 10, 2)]
])
def test_log(shapes):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data)
        torch_data = torch.tensor(np_data)

        dl_out = dlgrad_data.log()
        to_out = torch_data.log()
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)
