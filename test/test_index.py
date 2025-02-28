import numpy as np
import pytest
import torch
from dlgrad import Tensor


# Thanks to tinygrad for the template
def run(dlgrad_data, torch_data, func):
    np.testing.assert_allclose(func(*dlgrad_data).numpy(), func(*torch_data).numpy(), atol=1e-6, rtol=1e-3)

@pytest.mark.parametrize("shapes", [
    [(6, 3, 4), (6, 3, 4)],
    [(6, 3), (6, 3)]
])
def test(shapes):
    for sh in shapes:
        np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
        dlgrad_data = [Tensor(data) for data in np_data]
        torch_data = [torch.tensor(data) for data in np_data]

        dlgrad_data = [i[2:4] for i in dlgrad_data]
        torch_data = [i[2:4] for i in torch_data]

        run(dlgrad_data, torch_data, lambda x, y: x+y)
        run(dlgrad_data, torch_data, lambda x, y: x-y)
        run(dlgrad_data, torch_data, lambda x, y: x*y)
        run(dlgrad_data, torch_data, lambda x, y: x/y)

        np.testing.assert_allclose(dlgrad_data[0].sum().numpy(), dlgrad_data[0].sum().numpy(), atol=1e-6, rtol=1e-3)
        np.testing.assert_allclose(dlgrad_data[0].sum(0).numpy(), dlgrad_data[0].sum(0).numpy(), atol=1e-6, rtol=1e-3)
        np.testing.assert_allclose(dlgrad_data[0].sum(1).numpy(), dlgrad_data[0].sum(1).numpy(), atol=1e-6, rtol=1e-3)
        np.testing.assert_allclose(dlgrad_data[0].max().numpy(), dlgrad_data[0].max().numpy(), atol=1e-6, rtol=1e-3)
        np.testing.assert_allclose(dlgrad_data[0].max(0).numpy(), dlgrad_data[0].max(0).numpy(), atol=1e-6, rtol=1e-3)
        np.testing.assert_allclose(dlgrad_data[0].max(1).numpy(), dlgrad_data[0].max(1).numpy(), atol=1e-6, rtol=1e-3)




        
