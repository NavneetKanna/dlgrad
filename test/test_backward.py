import numpy as np
import pytest
import torch
from dlgrad import Tensor
from itertools import product
import torch.nn.functional as F


@pytest.fixture(params=['cpu', 'metal'])
def device(request):
    if request.param == 'metal' and not torch.backends.mps.is_available():
        pytest.skip("Apple Metal GPU not available")
    return request.param

def to_torch_device(device):
    return "mps" if device == "metal" else device

def run_binary_op_backward(shapes, device, op):
    np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
    dl_x, dl_y = [Tensor(data, requires_grad=True) for data in np_data]
    th_x, th_y = [torch.tensor(data, device=to_torch_device(device), requires_grad=True) for data in np_data]

    op(dl_x, dl_y).sum().backward()
    op(th_x, th_y).sum().backward()

    np.testing.assert_allclose(
        dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3
    )
    np.testing.assert_allclose(
        dl_y.grad.numpy(), th_y.grad.cpu().numpy(), atol=1e-6, rtol=1e-3
    )

def run_unary_op_backward(shape, device, op):
    np_data = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)
    dl_x = Tensor(np_data, requires_grad=True)
    th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    op(dl_x).sum().backward()
    op(th_x).sum().backward()

    np.testing.assert_allclose(
        dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3
    )

def generate_broadcastable_shapes(shape):
    options = [(1, d) for d in shape]
    return [(shape, dims) for dims in product(*options)]

shapes = (
    generate_broadcastable_shapes((2, 3, 4, 5))
    + generate_broadcastable_shapes((2, 3, 4))
    + generate_broadcastable_shapes((784, 64))
)

@pytest.mark.parametrize("shapes", shapes)
@pytest.mark.parametrize("op", [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / y])
def test_elementwise_backward(shapes, device, op):
    run_binary_op_backward(shapes, device, op)

@pytest.mark.parametrize("shape", [(4,3,2,4), (4,3,2), (3,2)])
@pytest.mark.parametrize("op", [
    lambda x: x.relu(),
    lambda x: x.leaky_relu() if not isinstance(x, torch.Tensor) else F.leaky_relu(x),
    lambda x: x.tanh(),
    lambda x: x.sigmoid(),
    lambda x: x.sum(),
])
def test_unary_backward(shape, device, op):
    run_unary_op_backward(shape, device, op)

@pytest.mark.parametrize("shape", [(5, 4, 3, 2), (2, 3, 4), (2, 3)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_max_backward(shape, keepdims, device):
    for dim in range(len(shape)):
        np_data = np.random.randn(*shape).astype(np.float32)
        dl_x = Tensor(np_data, device=device, requires_grad=True)
        th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

        dl_x.max(dim=dim, keepdim=keepdims).sum().backward()
        o, _ = th_x.max(dim=dim, keepdim=keepdims)
        o.sum().backward()

        np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

    np_data = np.random.randn(*shape).astype(np.float32)
    dl_x = Tensor(np_data, device=device, requires_grad=True)
    th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    dl_x.max().backward()
    th_x.max().backward()

    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shape", [(5, 4, 3, 2), (2, 3, 4), (2, 3)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_sum_backward(shape, keepdims, device):
    for dim in range(len(shape)):
        np_data = np.random.randn(*shape).astype(np.float32)
        dl_x = Tensor(np_data, requires_grad=True)
        th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

        dl_x.sum(dim=dim, keepdim=keepdims).sum().backward()
        th_x.sum(dim=dim, keepdim=keepdims).sum().backward()

        np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

    np_data = np.random.randn(*shape).astype(np.float32)
    dl_x = Tensor(np_data, requires_grad=True)
    th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    dl_x.sum().backward()
    th_x.sum().backward()

    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shape", [(3,2,4), (4,4)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_mean_backward(shape, keepdims, device):
    for dim in range(len(shape)):
        np_data = np.random.randn(*shape).astype(np.float32)
        dl_x = Tensor(np_data, requires_grad=True)
        th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

        dl_x.mean(dim=dim, keepdim=keepdims).sum().backward()
        th_x.mean(dim=dim, keepdim=keepdims).sum().backward()

        np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

    np_data = np.random.randn(*shape).astype(np.float32)
    dl_x = Tensor(np_data, requires_grad=True)
    th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    dl_x.mean().backward()
    th_x.mean().backward()

    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    ((2,3), (3,4)),
    ((4,8), (8,16)),
    ((128,64), (64,10)),
])
def test_matmul_backward(shapes, device):
    x_shape, y_shape = shapes
    np_x = np.random.randn(*x_shape).astype(np.float32)
    np_y = np.random.randn(*y_shape).astype(np.float32)
    dl_x = Tensor(np_x, requires_grad=True)
    dl_y = Tensor(np_y, requires_grad=True)
    th_x = torch.tensor(np_x, device=to_torch_device(device), requires_grad=True)
    th_y = torch.tensor(np_y, device=to_torch_device(device), requires_grad=True)

    dl_out = dl_x @ dl_y
    th_out = th_x @ th_y
    dl_out.sum().backward()
    th_out.sum().backward()

    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(dl_y.grad.numpy(), th_y.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [[(2, 3)]])
def test_cross_entropy_backward(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        np_target = np.array([[1, 2]]).astype(np.float32).reshape(-1, 1)

        dl_x = Tensor(np_data, device=device, requires_grad=True)
        dl_t = Tensor(np_target)
        th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)
        th_t = torch.tensor(np_target, dtype=torch.long).squeeze()

        dl_x.cross_entropy_loss(dl_t).backward()
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
        loss(th_x, th_t).backward()

        np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [[(2, 3)]])
def test_log_softmax_backward(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dl_x = Tensor(np_data, device=device, requires_grad=True)

        th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

        dl_x.log_softmax(dim=1).sum().backward()
        to_out = torch.nn.LogSoftmax(dim=1)
        to_out(th_x).sum().backward()

        np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)
