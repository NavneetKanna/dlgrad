import numpy as np
import pytest
import torch
from dlgrad.tensor import Tensor

ce_shapes = [
    [(4, 7)],
]

@pytest.mark.parametrize("shapes", ce_shapes)
def test_cross_entropy(shapes):
    for sh in shapes:
        np_logits = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        np_target = np.random.randint(sh[0], size=(sh[0], 1)).astype(np.float32)

        dl_logits = Tensor(np_logits, requires_grad=True)
        dl_target = Tensor(np_target)

        to_logits = torch.tensor(np_logits, requires_grad=True)
        to_target = torch.tensor(np_target, dtype=torch.long).squeeze()

        dl_out = dl_logits.cross_entropy_loss(dl_target)

        to_crit = torch.nn.CrossEntropyLoss(reduction='sum')
        to_out = to_crit(to_logits, to_target)

        np.testing.assert_allclose(dl_out.numpy(), to_out.detach().numpy(), atol=1e-6, rtol=1e-4)

        dl_out.backward()
        to_out.backward()

        dl_grad = dl_logits.grad.numpy()
        to_grad = to_logits.grad
        np.testing.assert_allclose(dl_grad, to_grad.detach().numpy(), atol=1e-5, rtol=1e-4)

ce_shapes = [
    [(4, 1)],
]
@pytest.mark.parametrize("shapes", ce_shapes)
def test_bcewithlogitsloss(shapes):
    for sh in shapes:
        np_logits = np.random.uniform(low=-1.0, high=1.0, size=sh).astype(np.float32)
        np_target = np.random.randint(sh[0], size=(sh[0], 1)).astype(np.float32)

        dl_logits = Tensor(np_logits, requires_grad=True)
        dl_target = Tensor(np_target)

        to_logits = torch.tensor(np_logits, requires_grad=True)
        to_target = torch.tensor(np_target)

        dl_out = dl_logits.bcewithlogitsloss(dl_target)

        to_crit = torch.nn.BCEWithLogitsLoss()
        to_out = to_crit(to_logits, to_target)

        np.testing.assert_allclose(dl_out.numpy(), to_out.detach().numpy(), atol=1e-6, rtol=1e-4)

        dl_out.backward()
        to_out.backward()

        dl_grad = dl_logits.grad.numpy()
        to_grad = to_logits.grad
        np.testing.assert_allclose(dl_grad, to_grad.detach().numpy(), atol=1e-5, rtol=1e-4)
