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
def test_matmul_2d_backward(shapes, device):
    x_shape, y_shape = shapes
    np_x = np.random.rand(*x_shape).astype(np.float32)
    np_y = np.random.rand(*y_shape).astype(np.float32)
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

@pytest.mark.parametrize("shapes", [
    ((2, 3, 4), (2, 4, 5)),
    ((100, 200, 300), (100, 300, 200)),
    ((100, 200, 300), (1, 300, 200)),
    ((1, 200, 300), (100, 300, 200)),
])
def test_matmul_3d_backward(shapes, device):
    x_shape, y_shape = shapes
    np_x = np.random.rand(*x_shape).astype(np.float32)
    np_y = np.random.rand(*y_shape).astype(np.float32)
    dl_x = Tensor(np_x, requires_grad=True)
    dl_y = Tensor(np_y, requires_grad=True)
    th_x = torch.tensor(np_x, device=to_torch_device(device), requires_grad=True)
    th_y = torch.tensor(np_y, device=to_torch_device(device), requires_grad=True)

    dl_out = dl_x @ dl_y
    th_out = th_x @ th_y
    dl_out.sum().backward()
    th_out.sum().backward()

    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().detach().numpy(), atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(dl_y.grad.numpy(), th_y.grad.cpu().detach().numpy(), atol=1e-6, rtol=1e-3)

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
def test_matmul_4d_backward(shapes, device):
    x_shape, y_shape = shapes
    np_x = np.random.rand(*x_shape).astype(np.float32)
    np_y = np.random.rand(*y_shape).astype(np.float32)
    dl_x = Tensor(np_x, requires_grad=True)
    dl_y = Tensor(np_y, requires_grad=True)
    th_x = torch.tensor(np_x, device=to_torch_device(device), requires_grad=True)
    th_y = torch.tensor(np_y, device=to_torch_device(device), requires_grad=True)

    dl_out = dl_x @ dl_y
    th_out = th_x @ th_y
    dl_out.sum().backward()
    th_out.sum().backward()

    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().detach().numpy(), atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(dl_y.grad.numpy(), th_y.grad.cpu().detach().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [[(2, 3)]])
def test_cross_entropy_backward(shapes, device):
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        np_target = np.array([[1, 2]]).astype(np.float32).reshape(-1, 1)

        dl_x = Tensor(np_data, device=device, requires_grad=True)
        dl_t = Tensor(np_target)
        th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)
        th_t = torch.tensor(np_target, device=to_torch_device(device), dtype=torch.long).squeeze()

        dl_x.cross_entropy_loss(dl_t).backward()
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
        loss(th_x, th_t).backward()

        np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

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

@pytest.mark.parametrize("shape", [
    (4, 3),
    (100, 200),
])
def test_transpose_2d_backward(shape, device):
    np_data = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)
    dl_x = Tensor(np_data, device=device, requires_grad=True)
    th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    dl_x.T.sum().backward()
    th_x.T.sum().backward()
    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

    dl_x.grad = None
    th_x.grad = None

    dl_x.T.sum().backward()
    th_x.T.sum().backward()
    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shape", [
    (4, 3, 5),
    (10, 20, 30),
])
def test_transpose_3d_backward(shape, device):
    np_data = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)
    dl_x = Tensor(np_data, device=device, requires_grad=True)
    th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    dl_x.transpose(0, 1).sum().backward()
    th_x.transpose(0, 1).sum().backward()
    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

    dl_x.grad = None
    th_x.grad = None

    dl_x.transpose(1, 2).sum().backward()
    th_x.transpose(1, 2).sum().backward()
    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

shapes = [
    (4, 3, 5, 6),
    (10, 20, 30, 40)
]
@pytest.mark.parametrize("shape", shapes)
def test_transpose_4d_backward(shape, device):
    np_data = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)
    dl_x = Tensor(np_data, device=device, requires_grad=True)
    th_x = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    dl_x.transpose(1, 2).sum().backward()
    th_x.transpose(1, 2).sum().backward()
    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

    dl_x.grad = None
    th_x.grad = None

    dl_x.transpose(2, 3).sum().backward()
    th_x.transpose(2, 3).sum().backward()
    np.testing.assert_allclose(dl_x.grad.numpy(), th_x.grad.cpu().numpy(), atol=1e-6, rtol=1e-3)

shapes = [(4, 3, 2, 4), (4, 3, 2), (3, 2)]
@pytest.mark.parametrize("shape", shapes)
def test_where(shape, device):
    np_data = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)
    da = Tensor(np_data, device=device, requires_grad=True)
    ta = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    db = Tensor.where(da>0.0, 0.0, da)
    tb = torch.where(ta>0.0, 0.0, ta)

    db.sum().backward()
    tb.sum().backward()

    np.testing.assert_allclose(da.grad.numpy(), ta.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)

    da.grad = None
    ta.grad = None

    db = Tensor.where(da>0.0, da, 0.0)
    tb = torch.where(ta>0.0, ta, 0.0)

    db.sum().backward()
    tb.sum().backward()

    np.testing.assert_allclose(da.grad.numpy(), ta.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)

shapes = [(4, 3, 2, 4), (4, 3, 2)]
@pytest.mark.parametrize("shape", shapes)
def test_masked_fill(shape, device):
    np_data = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)
    da = Tensor(np_data, device=device, requires_grad=True)
    ta = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    mask = da > 0.0
    t_mask = ta > 0.0

    db = da.masked_fill(mask, 0.0)
    tb = ta.masked_fill(t_mask, 0.0)

    db.sum().backward()
    tb.sum().backward()

    np.testing.assert_allclose(da.grad.numpy(), ta.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)

    da.grad = None
    ta.grad = None

    fill_val = -1.5
    db = da.masked_fill(mask, fill_val)
    tb = ta.masked_fill(t_mask, fill_val)

    db.sum().backward()
    tb.sum().backward()

    np.testing.assert_allclose(da.grad.numpy(), ta.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)

shapes = [(4, 4), (3, 4), (2, 3)]
@pytest.mark.parametrize("shape", shapes)
def test_tril(shape, device):
    np_data = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32)
    da = Tensor(np_data, device=device, requires_grad=True)
    ta = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    # Case 1: Default diagonal (0) - Main diagonal and below
    db = Tensor.tril(da, k=0.0)
    tb = torch.tril(ta, diagonal=0)

    db.sum().backward()
    tb.sum().backward()

    np.testing.assert_allclose(da.grad.numpy(), ta.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)

    # Reset gradients
    da.grad = None
    ta.grad = None

    # Case 2: Offset diagonal (1) - Main diagonal, 1 above, and everything below
    # This verifies the logic handles 'k' correctly
    db = Tensor.tril(da, k=1.0)
    tb = torch.tril(ta, diagonal=1)

    db.sum().backward()
    tb.sum().backward()

    np.testing.assert_allclose(da.grad.numpy(), ta.grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("p", [0.0, 1.0])
def test_dropout_deterministic(p, device):
    shape = (10, 10)
    np_data = np.random.randn(*shape).astype(np.float32)

    da = Tensor(np_data, device=device, requires_grad=True)
    ta = torch.tensor(np_data, device=to_torch_device(device), requires_grad=True)

    db = da.dropout(p=p)
    tb = torch.nn.functional.dropout(ta, p=p)

    db.sum().backward()
    tb.sum().backward()

    np.testing.assert_allclose(da.grad.numpy(), ta.grad.detach().cpu().numpy(), atol=1e-5)

