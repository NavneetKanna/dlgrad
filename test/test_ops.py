import numpy as np
import pytest
import torch

from dlgrad import Tensor


# TODO: Tets NaN's

# Thanks to tinygrad for the template
def run(shapes: list[tuple], func):
    np_data = [np.random.uniform(size=sh).astype(np.float32) for sh in shapes]
    dlgrad_data = [Tensor(data) for data in np_data]
    torch_data = [torch.tensor(data) for data in np_data]

    np.testing.assert_allclose(func(*dlgrad_data).numpy(), func(*torch_data).numpy(), atol=1e-6, rtol=1e-3)
    
@pytest.mark.parametrize("shapes", [
    [(2, 3), (2, 3)],
    [(100, 100), (100, 100)],
    [(78, 91), (78, 91)],
    [(4, 3, 2), (4, 3, 2)],
    [(87, 3, 10), (87, 3, 10)],
])
def test_add_same_shape(shapes):
    run(shapes, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (1, 3)],
    [(2, 3), (2, 1)],
    [(4, 3, 2), (3, 2)],
    [(4, 3, 2), (1, 2)],
    [(4, 3, 2), (3, 1)],
])
def test_add_diff_shape(shapes):
    run(shapes, lambda x, y: x+y)
    