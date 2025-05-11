import numpy as np
import pytest
import torch
from dlgrad import Tensor

# TODO: Test NaN's

# Thanks to tinygrad for the template
def run(shapes: list[tuple], func):
    np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
    dlgrad_data = [Tensor(data, device="metal") for data in np_data]
    torch_data = [torch.tensor(data, device="mps") for data in np_data]

    np.testing.assert_allclose(func(*dlgrad_data).numpy(), func(*torch_data).cpu().numpy(), atol=1e-6, rtol=1e-3)


s = [
    [(64, 64), (64, 64)],
    [(4096, 4096), (4096, 4096)],
    [(64, 65), (64, 65)],
]

@pytest.mark.parametrize("shapes", s)
def test_add(shapes):
    run(shapes, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", s)
def test_sub(shapes):
    run(shapes, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", s)
def test_mul(shapes):
    run(shapes, lambda x, y: x*y)

@pytest.mark.parametrize("shapes", s)
def test_div(shapes):
    run(shapes, lambda x, y: x/y)

@pytest.mark.parametrize("shapes", [
    [(4096, 4096), (4096, 4096)],
    [(64, 65), (65, 64)],
])
def test_matmul(shapes):
    run(shapes, lambda x, y: x@y)

@pytest.mark.parametrize("shapes", [
    [(64, 64)],
    [(4096, 4096)],
    [(64, 65)],
    [(65, 64)],
])
def test_transpose_same_tensors(shapes):
    run(shapes, lambda x: x@x.T)

@pytest.mark.parametrize("shapes", [
    [(64, 64)]
])
def test_pow(shapes):
    run(shapes, lambda x: x**2)

@pytest.mark.parametrize("shapes", [
    [(64, 64)],
    [(4096, 4096)],
    [(64, 65)],
    [(65, 64)],
])
def test_sum_2d(shapes):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device="metal")
        torch_data = torch.tensor(np_data, device="mps")

        dl_out = dlgrad_data.sum(dim=0)
        to_out = torch_data.sum(dim=0, keepdim=True)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.sum(dim=1)
        to_out = torch_data.sum(dim=1, keepdim=True)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.sum()
        to_out = torch_data.sum()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
     [(64, 64)],
    [(4096, 4096)],
    [(64, 65)],
    [(65, 64)],
])
def test_max_2d(shapes):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device="metal")
        torch_data = torch.tensor(np_data, device="mps")

        dl_out = dlgrad_data.max(dim=0)
        to_out, _ = torch_data.max(dim=0, keepdim=True)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.max(dim=1)
        to_out, _ = torch_data.max(dim=1, keepdim=True)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.max()
        to_out = torch_data.max()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_exp(shapes):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device="metal")
        torch_data = torch.tensor(np_data, device="mps")

        dl_out = dlgrad_data.exp()
        to_out = torch_data.exp()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_log(shapes):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device="metal")
        torch_data = torch.tensor(np_data, device="mps")

        dl_out = dlgrad_data.log()
        to_out = torch_data.log()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_sqrt(shapes):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device="metal")
        torch_data = torch.tensor(np_data, device="mps")

        dl_out = dlgrad_data.sqrt()
        to_out = torch_data.sqrt()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)
