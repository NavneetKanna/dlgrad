import numpy as np
import pytest
import torch
from dlgrad import Tensor
from itertools import product

# TODO: Test NaN's
# TODO: Check ops with (1,)

@pytest.fixture(params=['cpu', 'metal'])
def device(request):
    if request.param == 'metal' and not torch.backends.mps.is_available():
        pytest.skip("Apple Metal GPU not available")
    return request.param

# Thanks to tinygrad for the template
def run(shapes: list[tuple], device, func):
    np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
    dlgrad_data = [Tensor(data, device=device) for data in np_data]
    if device == "metal":
        device = "mps"
    torch_data = [torch.tensor(data, device=device) for data in np_data]

    dlgrad_result = func(*dlgrad_data)
    torch_result = func(*torch_data)
    if device == "mps":
        torch_result = torch_result.cpu()

    np.testing.assert_allclose(dlgrad_result.numpy(), torch_result.numpy(), atol=1e-6, rtol=1e-3)

def generate_broadcastable_shapes(shape, reverse: bool = False):
    options = [(1, d) for d in shape]
    result = []
    for dims in product(*options):
        if dims == shape:
            if not reverse:
                result.append((shape, dims))
            else:
                result.append((dims, shape))
        else:
            if not reverse:
                result.append((shape, dims))
            else:
                result.append((dims, shape))
    return result

s = generate_broadcastable_shapes((2, 3, 4, 5))
s += generate_broadcastable_shapes((2, 3, 4))
s += generate_broadcastable_shapes((2, 3))

rs = generate_broadcastable_shapes((2, 3, 4, 5), reverse=True)
rs += generate_broadcastable_shapes((2, 3, 4), reverse=True)
rs += generate_broadcastable_shapes((2, 3), reverse=True)

@pytest.mark.parametrize("shapes", s)
def test_add(shapes, device):
    run(shapes, device, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", rs)
def test_add_diff_shape_reverse(shapes, device):
    if device == 'metal':
        pytest.skip()
    run(shapes, device, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", s)
def test_sub(shapes, device):
    run(shapes, device, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", rs)
def test_sub_diff_shape_reverse(shapes, device):
    if device == 'metal':
        pytest.skip()
    run(shapes, device, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", s)
def test_mul(shapes, device):
    run(shapes, device, lambda x, y: x*y)

@pytest.mark.parametrize("shapes", rs)
def test_mul_diff_shape_reverse(shapes, device):
    if device == 'metal':
        pytest.skip()
    run(shapes, device, lambda x, y: x*y)

@pytest.mark.parametrize("shapes", s)
def test_div(shapes, device):
    run(shapes, device, lambda x, y: x/y)

@pytest.mark.parametrize("shapes", rs)
def test_div_diff_shape_reverse(shapes, device):
    if device == 'metal':
        pytest.skip()
    run(shapes, device, lambda x, y: x/y)

@pytest.mark.parametrize("shapes", [[(2, 3), (3, 2)], [(20, 20), (20, 20)]])
def test_matmul(shapes, device):
    # if device == 'metal':
        # pytest.skip()
    run(shapes, device, lambda x, y: x@y)

@pytest.mark.parametrize("shapes", [[(2, 3), (2, 3)]])
def test_transpose_diff_tensors(shapes, device):
    if device == 'metal':
        pytest.skip()
    run(shapes, device, lambda x, y: x@y.T)

@pytest.mark.parametrize("shapes", [[(2, 3), (2, 3)]])
def test_transpose_diff_tensors_reverse(shapes, device):
    if device == 'metal':
        pytest.skip()
    run(shapes, device, lambda x, y: x.T@y)

@pytest.mark.parametrize("shapes", [[(4, 3)]])
def test_transpose_same_tensors(shapes, device):
    if device == 'metal':
        pytest.skip()
    run(shapes, device, lambda x: x@x.T)

@pytest.mark.parametrize("shapes", [[(4, 3, 2, 4)], [(4, 3, 2)], [(4, 3)]])
def test_pow(shapes, device):
    if device == 'metal':
        pytest.skip()
    
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        dl_out = dlgrad_data**2
        to_out = torch_data**2
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)
    
# TODO: Sum 1d ?
@pytest.mark.parametrize("shapes", [[(4, 3, 2, 4)], [(4, 3, 2)], [(4, 3)], [(1000, 1000)]])
def test_sum(shapes, device):
    # if device == 'metal':
        # pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        for dim in range(len(sh)):
            dl_out = dlgrad_data.sum(dim=dim)
            to_out = torch_data.sum(dim=dim)
            if device == "mps":
                to_out = to_out.cpu()

            np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [[(4, 3, 2, 4)], [(4, 3, 2)], [(4, 3)]])
def test_mean(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        torch_data = torch.tensor(np_data, device=device)

        for dim in range(len(sh)):
            dl_out = dlgrad_data.mean(dim=dim)
            to_out = torch_data.mean(dim=dim)

            np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes",[[(4, 3, 2, 4)], [(4, 3, 2)], [(3, 2)], [(1000, 1000)]])
def test_max(shapes, device):
    # if device == 'metal':
        # pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        for dim in range(len(sh)):
            dl_out = dlgrad_data.max(dim=dim)
            to_out, _ = torch_data.max(dim=dim)
            if device == "mps":
                to_out = to_out.cpu()

            np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

s = [[(4, 3, 2, 4)], [(4, 3, 2)], [(3, 2)]]

@pytest.mark.parametrize("shapes", s)
def test_where(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        dl_out = Tensor.where(dlgrad_data>0.0, 1.0, 0.0)
        to_out = torch.where(torch_data>0.0, 1.0, 0.0)
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        dl_out = Tensor.where(dlgrad_data>0.0, dlgrad_data, 0.0)
        to_out = torch.where(torch_data>0.0, torch_data, 0.0)
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_relu(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        dl_out = dlgrad_data.relu()
        to_out = torch_data.relu()
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_leaky_relu(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        dl_out = dlgrad_data.leaky_relu()
        m = torch.nn.LeakyReLU()
        to_out = m(torch_data)
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_sigmoid(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        dl_out = dlgrad_data.sigmoid()
        to_out = torch_data.sigmoid()
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_tanh(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        dl_out = dlgrad_data.tanh()
        to_out = torch_data.tanh()
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_exp(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        dl_out = dlgrad_data.exp()
        to_out = torch_data.exp()
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_log(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        dl_out = dlgrad_data.log()
        to_out = torch_data.log()
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_sqrt(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        dl_out = dlgrad_data.sqrt()
        to_out = torch_data.sqrt()
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", s)
def test_logsoftmax(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device)

        dl_out = dlgrad_data.log_softmax()
        m = torch.nn.LogSoftmax(dim=1)
        to_out = m(torch_data)
        if device == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

s = [
    [(4, 3, 2, 1)],
    [(4, 3, 2)]
]
@pytest.mark.parametrize("shapes", s)
def test_transpose(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=device)

        dl_out = Tensor.transpose(dlgrad_data, (0, 1))
        to_out = torch.transpose(torch_data, 0, 1)
        np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

        if len(sh) == 4:
            dl_out = Tensor.transpose(dlgrad_data, (0, 3))
            to_out = torch.transpose(torch_data, 0, 3)
            np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

            dl_out = Tensor.transpose(dlgrad_data, (1, 2))
            to_out = torch.transpose(torch_data, 1, 2)
            np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

            dl_out = Tensor.transpose(dlgrad_data, (2, 3))
            to_out = torch.transpose(torch_data, 2, 3)
            np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)
        elif len(sh) == 3:
            dl_out = Tensor.transpose(dlgrad_data, (1, 2))
            to_out = torch.transpose(torch_data, 1, 2)
            np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

            dl_out = Tensor.transpose(dlgrad_data, (0, 2))
            to_out = torch.transpose(torch_data, 0, 2)
            np.testing.assert_allclose(dl_out.numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

