import numpy as np
import pytest
import torch
from dlgrad.tensor import Tensor

@pytest.fixture(params=['cpu', 'metal'])
def device(request):
    if request.param == 'metal' and not torch.backends.mps.is_available():
        pytest.skip("Apple Metal GPU not available")
    return request.param

def to_torch_device(device):
    return "mps" if device == "metal" else device

@pytest.mark.parametrize("shapes", [(4, 7), (1000, 1000)])
def test_cross_entropy(shapes, device):
    np_logits = np.random.uniform(low=-1.0, high=1.0, size=shapes).astype(np.float32)
    np_target = np.random.randint(shapes[0], size=(shapes[0], 1)).astype(np.float32)

    dl_logits = Tensor(np_logits, device=device, requires_grad=True)
    dl_target = Tensor(np_target, device=device)

    to_logits = torch.tensor(np_logits, device=to_torch_device(device), requires_grad=True)
    to_target = torch.tensor(np_target, device=to_torch_device(device), dtype=torch.long).squeeze()

    dl_out = dl_logits.cross_entropy_loss(dl_target)

    to_crit = torch.nn.CrossEntropyLoss(reduction='sum')
    to_out = to_crit(to_logits, to_target)

    np.testing.assert_allclose(dl_out.numpy(), to_out.detach().cpu().numpy(), atol=1e-6, rtol=1e-4)

    dl_out.backward()
    to_out.backward()

    dl_grad = dl_logits.grad.numpy()
    to_grad = to_logits.grad

    np.testing.assert_allclose(dl_grad, to_grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-4)

@pytest.mark.parametrize("shapes", [(4, 7), (1000, 1000)])
def test_bcewithlogitsloss(shapes, device):
    np_logits = np.random.uniform(low=-1.0, high=1.0, size=shapes).astype(np.float32)
    np_target = np.random.randint(2, size=shapes).astype(np.float32)

    dl_logits = Tensor(np_logits, device=device, requires_grad=True)
    dl_target = Tensor(np_target, device=device)

    to_logits = torch.tensor(np_logits, device=to_torch_device(device), requires_grad=True)
    to_target = torch.tensor(np_target, device=to_torch_device(device))

    dl_out = dl_logits.bcewithlogitsloss(dl_target)

    to_crit = torch.nn.BCEWithLogitsLoss()
    to_out = to_crit(to_logits, to_target)

    np.testing.assert_allclose(dl_out.numpy(), to_out.detach().cpu().numpy(), atol=1e-6, rtol=1e-4)

    dl_out.backward()
    to_out.backward()

    dl_grad = dl_logits.grad.numpy()
    to_grad = to_logits.grad

    np.testing.assert_allclose(dl_grad, to_grad.detach().cpu().numpy(), atol=1e-5, rtol=1e-4)
