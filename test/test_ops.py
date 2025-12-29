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

def to_torch_device(device):
    return "mps" if device == "metal" else device

def run(shapes: list[tuple], device, func):
    np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
    dlgrad_data = [Tensor(data, device=device) for data in np_data]
    torch_data = [torch.tensor(data, device=to_torch_device(device)) for data in np_data]

    dlgrad_result = func(*dlgrad_data)
    torch_result = func(*torch_data)

    np.testing.assert_allclose(dlgrad_result.numpy(), torch_result.cpu().numpy(), atol=1e-6, rtol=1e-3)

def generate_broadcastable_shapes(shape):
    options = [(1, d) for d in shape]
    return [(shape, dims) for dims in product(*options)]

shapes = (
    generate_broadcastable_shapes((2, 3, 4, 5))
    + generate_broadcastable_shapes((2, 3, 4))
    + generate_broadcastable_shapes((784, 64))
)

reverse_shapes = (
    generate_broadcastable_shapes((2, 3, 4, 5))
    + generate_broadcastable_shapes((2, 3, 4))
    + generate_broadcastable_shapes((784, 64))
)

@pytest.mark.parametrize("shapes", shapes)
def test_add(shapes, device):
    run(shapes, device, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", reverse_shapes)
def test_add_diff_shape_reverse(shapes, device):
    run(shapes, device, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", shapes)
def test_sub(shapes, device):
    run(shapes, device, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", reverse_shapes)
def test_sub_diff_shape_reverse(shapes, device):
    run(shapes, device, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", shapes)
def test_mul(shapes, device):
    run(shapes, device, lambda x, y: x*y)

@pytest.mark.parametrize("shapes", reverse_shapes)
def test_mul_diff_shape_reverse(shapes, device):
    run(shapes, device, lambda x, y: x*y)

@pytest.mark.parametrize("shapes", shapes)
def test_div(shapes, device):
    run(shapes, device, lambda x, y: x/y)

@pytest.mark.parametrize("shapes", reverse_shapes)
def test_div_diff_shape_reverse(shapes, device):
    run(shapes, device, lambda x, y: x/y)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (3, 2)],
    [(20, 20), (20, 20)],
    [(32, 8), (8, 16)],
    [(128, 784), (784, 64)],
    [(128, 64), (64, 10)],
    [(128, 10), (10, 64)],
    [(64, 128), (128, 10)],
    [(128, 64), (64, 784)],
    [(784, 128), (128, 64)],
    [(10000, 784), (784, 64)],
    [(10000, 64), (64, 10)]
])
def test_matmul(shapes, device):
    run(shapes, device, lambda x, y: x@y)

@pytest.mark.parametrize("shapes", [[(2, 3), (2, 3)]])
def test_transpose_diff_tensors(shapes, device):
    run(shapes, device, lambda x, y: x@y.T)

@pytest.mark.parametrize("shapes", [[(2, 3), (2, 3)]])
def test_transpose_diff_tensors_reverse(shapes, device):
    run(shapes, device, lambda x, y: x.T@y)

@pytest.mark.parametrize("shapes", [[(4, 3)]])
def test_transpose_same_tensors(shapes, device):
    run(shapes, device, lambda x: x@x.T)

@pytest.mark.parametrize("shapes", [[(4, 3, 2, 4)], [(4, 3, 2)], [(4, 3)]])
def test_pow(shapes, device):
    run(shapes, device, lambda x: x**2)

@pytest.mark.parametrize("shapes", [[(4, 3, 2, 4)], [(4, 3, 2)], [(4, 3)], [(1000, 1000)], [(4096, 4096)]])
def test_sum(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        for dim in range(len(sh)):
            dl_out = dlgrad_data.sum(dim=dim)
            to_out = torch_data.sum(dim=dim)
            np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.sum()
        to_out = torch_data.sum()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [[(4, 3, 2, 4)], [(4, 3, 2)], [(4, 3)]])
def test_mean(shapes, device):
    if device == 'metal':
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        for dim in range(len(sh)):
            dl_out = dlgrad_data.mean(dim=dim)
            to_out = torch_data.mean(dim=dim)
            np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [[(4, 3, 2, 4)], [(4, 3, 2)], [(3, 2)], [(1000, 1000)], [(4096, 4096)]])
def test_max(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        for dim in range(len(sh)):
            dl_out = dlgrad_data.max(dim=dim)
            to_out, _ = torch_data.max(dim=dim)
            np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.max()
        to_out = torch_data.max()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

shapes = [[(4, 3, 2, 4)], [(4, 3, 2)], [(3, 2)]]

@pytest.mark.parametrize("shapes", shapes)
def test_where(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = Tensor.where(dlgrad_data>0.0, 1.0, 0.0)
        to_out = torch.where(torch_data>0.0, 1.0, 0.0)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

        dl_out = Tensor.where(dlgrad_data>0.0, dlgrad_data, 0.0)
        to_out = torch.where(torch_data>0.0, torch_data, 0.0)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", shapes)
def test_relu(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.relu()
        to_out = torch_data.relu()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", shapes)
def test_leaky_relu(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.leaky_relu()
        m = torch.nn.LeakyReLU()
        to_out = m(torch_data)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", shapes)
def test_sigmoid(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.sigmoid()
        to_out = torch_data.sigmoid()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", shapes)
def test_tanh(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.tanh()
        to_out = torch_data.tanh()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", shapes)
def test_exp(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.exp()
        to_out = torch_data.exp()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", shapes)
def test_log(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=0.1, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.log()
        to_out = torch_data.log()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", shapes)
def test_sqrt(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.sqrt()
        to_out = torch_data.sqrt()
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", shapes)
def test_logsoftmax(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.log_softmax()
        m = torch.nn.LogSoftmax(dim=1)
        to_out = m(torch_data)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

shapes = [
    [(4, 3)],
    [(100, 200)]
]
@pytest.mark.parametrize("shapes", shapes)
def test_transpose(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.T
        to_out = torch_data.T
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

shapes = [
    [(4, 3, 5)],
    [(100, 200, 300)]
]
@pytest.mark.parametrize("shapes", shapes)
def test_transpose_3d(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.transpose(0, 1)
        to_out = torch_data.transpose(0, 1)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.transpose(1, 2)
        to_out = torch_data.transpose(1, 2)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

shapes = [
    [(4, 3, 5, 6)],
    [(10, 200, 300, 40)]
]
@pytest.mark.parametrize("shapes", shapes)
def test_transpose_4d(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, device=device)
        torch_data = torch.tensor(np_data, device=to_torch_device(device))

        dl_out = dlgrad_data.transpose(1, 2)
        to_out = torch_data.transpose(1, 2)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

        dl_out = dlgrad_data.transpose(2, 3)
        to_out = torch_data.transpose(2, 3)
        np.testing.assert_allclose(dl_out.numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(2, 3, 4), (2, 4, 3)],
    [(20, 20, 20), (20, 20, 20)],
    [(32, 8, 16), (32, 16, 8)],
    [(128, 784, 100), (128, 100, 10)],
    [(64, 128, 32), (64, 32, 64)],
    [(4000, 64, 50), (4000, 50, 70)],
    [(1, 784, 100), (2, 100, 10)],
    [(2, 784, 100), (1, 100, 10)],
])
def test_matmul_3d(shapes, device):
    run(shapes, device, lambda x, y: x@y)

@pytest.mark.parametrize("shapes", [
    [(2, 3, 4, 5), (2, 3, 5, 4)],
    [(20, 20, 20, 20), (20, 20, 20, 20)],
    [(32, 2, 8, 16), (32, 2, 16, 8)],
    [(128, 23, 784, 100), (128, 23, 100, 10)],
    [(64, 1, 128, 32), (64, 1, 32, 64)],
    [(64, 10, 128, 32), (64, 1, 32, 64)],
    [(64, 1, 128, 32), (64, 10, 32, 64)],
    #[(400, 2, 64, 50), (400, 2, 50, 70)],
    [(1, 1, 784, 100), (1, 1, 100, 10)],
    [(1, 784, 1, 100), (1, 784, 100, 1)],
])
def test_matmul_4d(shapes, device):
    run(shapes, device, lambda x, y: x@y)

shapes = [
    ((32, 10, 10), (32, 10, 10), 0.0),
    ((2, 8, 10, 10), (2, 8, 10, 10), -1.0),
    ((32, 10, 10), (1, 10, 10), 5.0),
    ((10, 10), (32, 10, 10), 99.0),
    ((32, 1, 10), (1, 10, 10), -5.0),
    ((2, 1, 10, 10), (1, 8, 10, 10), 7.0),
    ((4, 4, 4), (4, 4, 4), float('inf')),
    ((4, 4, 4), (4, 4, 4), float('-inf')),
    ((2, 4, 8, 8), (8, 8), float('-inf')),
]
@pytest.mark.parametrize("input_shape, mask_shape, fill_value", shapes)
def test_masked_fill(input_shape, mask_shape, fill_value, device):
    np_input = np.random.uniform(low=-10.0, high=10.0, size=input_shape).astype(np.float32)
    np_mask = (np.random.rand(*mask_shape) > 0.7).astype(np.float32)

    dl_x = Tensor(np_input, device=device)
    dl_mask = Tensor(np_mask, device=device)

    x_torch = torch.tensor(np_input, device=to_torch_device(device))
    mask_torch = torch.tensor(np_mask, device=to_torch_device(device)).bool()

    dl_out = dl_x.masked_fill(dl_mask, fill_value)
    out_torch = x_torch.masked_fill(mask_torch, fill_value)

    dl_res = dl_out.numpy()
    res_torch = out_torch.cpu().numpy()

    if fill_value == float('inf'):
        is_inf = np.isinf(res_torch) & (res_torch > 0)
        np.testing.assert_allclose(dl_res[~is_inf], res_torch[~is_inf], atol=1e-5, rtol=1e-5)
        assert np.all(dl_res[is_inf] > 1e37), "Filled values should be FLT_MAX (approx 3.4e38)"
    elif fill_value == float('-inf'):
        is_ninf = np.isneginf(res_torch)
        np.testing.assert_allclose(dl_res[~is_ninf], res_torch[~is_ninf], atol=1e-5, rtol=1e-5)
        assert np.all(dl_res[is_ninf] < -1e37), "Filled values should be -FLT_MAX (approx -3.4e38)"
    else:
        np.testing.assert_allclose(dl_res, res_torch, atol=1e-5, rtol=1e-5)

tril_shapes = [
        ((10, 10), 0.0),
        ((10, 10), 1.0),
        ((10, 10), -1.0),
        ((20, 10), 0.0),
        ((10, 20), 0.0),
        ((8, 8), 3.0),
        ((8, 8), -3.0),
]

@pytest.mark.parametrize("shape, diagonal", tril_shapes)
def test_tril_2d(shape, diagonal, device):
    np_input = np.random.uniform(low=-10.0, high=10.0, size=shape).astype(np.float32)

    x_custom = Tensor(np_input, device=device)

    x_torch = torch.tensor(np_input, device=to_torch_device(device))

    out_custom = Tensor.tril(x_custom, diagonal)

    out_torch = torch.tril(x_torch, diagonal=int(diagonal))

    dl_res = out_custom.numpy()
    res_torch = out_torch.cpu().numpy()

    np.testing.assert_allclose(dl_res, res_torch, atol=1e-5, rtol=1e-5)
