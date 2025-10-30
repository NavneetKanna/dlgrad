import numpy as np
import pytest
import torch
from dlgrad.tensor import Tensor

@pytest.fixture(params=['cpu', 'metal'])
def device(request):
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Metal GPU not available")
    return request.param

ce_shapes = [
    [(4, 7), (1000, 1000)],
]

@pytest.mark.parametrize("shapes", ce_shapes)
def test_cross_entropy(shapes, device):
    for sh in shapes:
        np_logits = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        np_target = np.random.randint(sh[0], size=(sh[0], 1)).astype(np.float32)

        dl_logits = Tensor(np_logits, device=device, requires_grad=True)
        dl_target = Tensor(np_target, device=device)

        if device == "metal":
            d = "mps"
        else:
            d = "cpu"

        to_logits = torch.tensor(np_logits, device=d, requires_grad=True)
        to_target = torch.tensor(np_target, device=d, dtype=torch.long).squeeze()

        dl_out = dl_logits.cross_entropy_loss(dl_target)

        to_crit = torch.nn.CrossEntropyLoss(reduction='sum')
        to_out = to_crit(to_logits, to_target)

        if d == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.detach().numpy(), atol=1e-6, rtol=1e-4)

        dl_out.backward()
        to_out.backward()

        dl_grad = dl_logits.grad.numpy()
        to_grad = to_logits.grad

        if d == "mps":
            to_grad = to_grad.cpu()

        np.testing.assert_allclose(dl_grad, to_grad.detach().numpy(), atol=1e-5, rtol=1e-4)

ce_shapes = [
    [(4, 1), (1000, 1000)],
]
@pytest.mark.parametrize("shapes", ce_shapes)
def test_bcewithlogitsloss(shapes, device):
    for sh in shapes:
        np_logits = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        np_target = np.random.randint(sh[0], size=(sh[0], 1)).astype(np.float32)

        dl_logits = Tensor(np_logits, device=device, requires_grad=True)
        dl_target = Tensor(np_target, device=device)

        if device == "metal":
            d = "mps"
        else:
            d = "cpu"

        to_logits = torch.tensor(np_logits, device=d, requires_grad=True)
        to_target = torch.tensor(np_target, device=d)

        dl_out = dl_logits.bcewithlogitsloss(dl_target)

        to_crit = torch.nn.BCEWithLogitsLoss()
        to_out = to_crit(to_logits, to_target)

        if d == "mps":
            to_out = to_out.cpu()

        np.testing.assert_allclose(dl_out.numpy(), to_out.detach().numpy(), atol=1e-6, rtol=1e-4)

        dl_out.backward()
        to_out.backward()

        dl_grad = dl_logits.grad.numpy()
        to_grad = to_logits.grad

        if d == "mps":
            to_grad = to_grad.cpu()

        np.testing.assert_allclose(dl_grad, to_grad.detach().numpy(), atol=1e-5, rtol=1e-4)
