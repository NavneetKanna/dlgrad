import numpy as np
import pytest
import torch
from dlgrad import Tensor


@pytest.fixture(params=['metal'])
def device(request):
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Metal GPU not available")
    return request.param

def run(dlgrad_data, torch_data, func, device):
    to_out = func(*torch_data)
    if device == "mps":
        to_out = to_out.cpu()
    np.testing.assert_allclose(func(*dlgrad_data).numpy(), to_out.numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(6, 3, 4), (6, 3, 4)],
    [(6, 3), (6, 3)]
])
def test(shapes, device):
        np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
        dlgrad_data = [Tensor(data, device=device) for data in np_data]
        if device == "metal":
            device = "mps"
        torch_data = [torch.tensor(data, device=device) for data in np_data]

        dlgrad_data = [i[2:4] for i in dlgrad_data]
        torch_data = [i[2:4] for i in torch_data]

        run(dlgrad_data, torch_data, lambda x, y: x+y, device)
        run(dlgrad_data, torch_data, lambda x, y: x-y, device)
        run(dlgrad_data, torch_data, lambda x, y: x*y, device)
        run(dlgrad_data, torch_data, lambda x, y: x/y, device)

        if device == "mps":
            torch_data = [i.cpu() for i in torch_data]

        np.testing.assert_allclose(dlgrad_data[0].sum().numpy(), torch_data[0].sum().numpy(), atol=1e-6, rtol=1e-3)
        np.testing.assert_allclose(dlgrad_data[0].sum(0).numpy(), torch_data[0].sum(0).numpy(), atol=1e-6, rtol=1e-3)
        np.testing.assert_allclose(dlgrad_data[0].sum(1).numpy(), torch_data[0].sum(1).numpy(), atol=1e-6, rtol=1e-3)
        # np.testing.assert_allclose(dlgrad_data[0].max().numpy(), torch_data[0].max().numpy(), atol=1e-6, rtol=1e-3)
        # np.testing.assert_allclose(dlgrad_data[0].max(0).numpy(), torch_data[0].max(0).numpy(), atol=1e-6, rtol=1e-3)
        # np.testing.assert_allclose(dlgrad_data[0].max(1).numpy(), torch_data[0].max(1).numpy(), atol=1e-6, rtol=1e-3)




        
