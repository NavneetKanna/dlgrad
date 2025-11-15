import numpy as np
import pytest
import torch
from dlgrad import Tensor

@pytest.fixture(params=['metal'])
def device(request):
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Metal GPU not available")
    return request.param

def to_torch_device(device):
    return "mps" if device == "metal" else device

def run(dlgrad_data, torch_data, func):
    to_out = func(*torch_data)
    np.testing.assert_allclose(func(*dlgrad_data).numpy(), to_out.cpu().numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(6, 3, 4), (6, 3, 4)],
    [(6, 3), (6, 3)]
])
def test(shapes, device):
    np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
    dlgrad_data = [Tensor(data, device=device) for data in np_data]
    torch_data = [torch.tensor(data, device=to_torch_device(device)) for data in np_data]

    dlgrad_data = [i[2:4] for i in dlgrad_data]
    torch_data = [i[2:4] for i in torch_data]

    run(dlgrad_data, torch_data, lambda x, y: x+y)
    run(dlgrad_data, torch_data, lambda x, y: x-y)
    run(dlgrad_data, torch_data, lambda x, y: x*y)
    run(dlgrad_data, torch_data, lambda x, y: x/y)

    np.testing.assert_allclose(dlgrad_data[0].sum().numpy(), torch_data[0].sum().cpu().numpy(), atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(dlgrad_data[0].sum(0).numpy(), torch_data[0].sum(0).cpu().numpy(), atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(dlgrad_data[0].sum(1).numpy(), torch_data[0].sum(1).cpu().numpy(), atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(dlgrad_data[0].max().numpy(), torch_data[0].max().cpu().numpy(), atol=1e-6, rtol=1e-3)
    o,_ = torch_data[0].max(0)
    np.testing.assert_allclose(dlgrad_data[0].max(0).numpy(), o.cpu().numpy(), atol=1e-6, rtol=1e-3)
    o,_ = torch_data[0].max(1)
    np.testing.assert_allclose(dlgrad_data[0].max(1).numpy(), o.cpu().numpy(), atol=1e-6, rtol=1e-3)
