import numpy as np
import torch
import pytest
from dlgrad import Tensor, nn

@pytest.fixture(params=["cpu", "metal"])
def device(request):
    if request.param == "metal" and not torch.backends.mps.is_available():
        pytest.skip("Metal not available")
    return request.param

def to_torch_device(device):
    return "mps" if device == "metal" else device

def test_linear(device):
    BS, in_dim, out_dim = 32, 8, 16

    dlgrad_model = nn.Linear(in_dim, out_dim, device=device)
    dlgrad_inp = Tensor.rand((BS, in_dim), device=device)
    dlgrad_out = dlgrad_model(dlgrad_inp)

    with torch.no_grad():
        torch_model = torch.nn.Linear(in_dim, out_dim, device=to_torch_device(device))
        torch_model.weight[:] = torch.tensor(dlgrad_model.weight.numpy(), device=to_torch_device(device), dtype=torch.float32)
        torch_model.bias[:] = torch.tensor(dlgrad_model.bias.numpy(), device=to_torch_device(device), dtype=torch.float32)

        torch_inp = torch.tensor(dlgrad_inp.numpy(), device=to_torch_device(device))
        torch_out = torch_model(torch_inp)

    np.testing.assert_allclose(dlgrad_out.numpy(), torch_out.cpu().numpy(), atol=5e-4, rtol=1e-5)
