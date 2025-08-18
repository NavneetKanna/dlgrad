import numpy as np
import pytest
import torch
from dlgrad import Tensor
from itertools import product


# TODO: Test with one tensor req grad is none

@pytest.fixture(params=['cpu', 'metal'])
def device(request):
    if request.param == 'metal' and not torch.backends.mps.is_available():
        pytest.skip("Apple Metal GPU not available")
    return request.param

# Thanks to tinygrad for the template
def run(shapes: list[tuple], device, func):
    np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
    dlgrad_data = [Tensor(data, requires_grad=True) for data in np_data]

    if device == "metal":
        device = "mps"
    torch_data = [torch.tensor(data, device=device, requires_grad=True) for data in np_data]

    func(*dlgrad_data).sum().backward()
    func(*torch_data).sum().backward()

    if device == "mps":
        torch_grad_1 = torch_data[0].grad.cpu()
        torch_grad_2 = torch_data[1].grad.cpu()
    else:
        torch_grad_1 = torch_data[0].grad
        torch_grad_2 = torch_data[1].grad

    np.testing.assert_allclose(dlgrad_data[0].grad.numpy(), torch_grad_1.numpy(), atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(dlgrad_data[1].grad.numpy(), torch_grad_2.numpy(), atol=1e-6, rtol=1e-3)

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

@pytest.mark.parametrize("shapes", s)
def test_add_backward(shapes, device):
    if device == 'metal' and any(len(shape) == 4 for shape in shapes):
        pytest.skip()
    run(shapes, device, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", s)
def test_sub_backward(shapes, device):
    if device == 'metal' and any(len(shape) == 4 for shape in shapes):
        pytest.skip()
    run(shapes, device, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", s)
def test_mul_backward(shapes, device):
    if device == 'metal' and any(len(shape) == 4 for shape in shapes):
        pytest.skip()
    run(shapes, device, lambda x, y: x*y)

@pytest.mark.parametrize("shapes", s)
def test_div_backward(shapes, device):
    if device == 'metal' and any(len(shape) == 4 for shape in shapes):
        pytest.skip()
    run(shapes, device,lambda x, y: x/y)

s = [[(4, 3, 2, 4)], [(4, 3, 2)], [(3, 2)]]

@pytest.mark.parametrize("shapes", s)
def test_relu_backward(shapes, device):
    if device == 'metal' and any(len(shape) == 4 for shape in shapes):
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, requires_grad=True)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device, requires_grad=True)

        dlgrad_data.relu().sum().backward()
        torch_data.relu().sum().backward()
        if device == "mps":
            torch_data = torch_data.grad.cpu()
        else:
            torch_data = torch_data.grad

        np.testing.assert_allclose(dlgrad_data.grad.numpy(), torch_data.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [[(4, 3, 2, 4)], [(4, 3, 2)], [(3, 2)]])
def test_max(shapes, device):
    if device == 'metal' and any(len(shape) == 4 for shape in shapes):
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)

        if device == "metal":
            device = "mps"

        for dim in range(len(sh)):
            dlgrad_data = Tensor(np_data, requires_grad=True)
            torch_data = torch.tensor(np_data, device=device, requires_grad=True)
            dlgrad_data.max(dim=dim).sum().backward()
            to_out, _ = torch_data.max(dim=dim)
            to_out.sum().backward()
            if device == "mps":
                torch_data = torch_data.grad.cpu()
            else:
                torch_data = torch_data.grad

            np.testing.assert_allclose(dlgrad_data.grad.numpy(), torch_data.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [[(2, 3)]])
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

@pytest.mark.parametrize("shapes", s)
def test_log_sft_backward(shapes, device):
    if device == 'metal' and any(len(shape) == 4 for shape in shapes) or any(len(shape) == 3 for shape in shapes):
        pytest.skip()
    for sh in shapes:
        np_data = np.random.uniform(size=sh).astype(np.float32)
        dlgrad_data = Tensor(np_data, requires_grad=True)

        if device == "metal":
            device = "mps"

        torch_data = torch.tensor(np_data, device=device, requires_grad=True)

        dlgrad_data.log_softmax(dim=1).sum().backward()
        to_out = torch.nn.LogSoftmax(dim=1)
        to_out(torch_data).sum().backward()
        if device == "mps":
            torch_data = torch_data.grad.cpu()
        else:
            torch_data = torch_data.grad

        np.testing.assert_allclose(dlgrad_data.grad.numpy(), torch_data.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [[(2, 3), (3, 2)]])
def test_matmul_backward(shapes, device):
    if device == 'metal' and len(shapes) == 4:
        pytest.skip()
    run(shapes, device, lambda x, y: x@y)
    