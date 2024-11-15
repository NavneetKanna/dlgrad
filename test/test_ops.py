import numpy as np
import pytest
import torch

from dlgrad import Tensor

# TODO: Test NaN's
# TOOD: Test Tensor(Scalar)

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

@pytest.mark.parametrize("shapes", [
    [(1, 3), (2, 3)],
    [(2, 1), (2, 3)],
    [(3, 2), (4, 3, 2)],
    [(1, 2), (4, 3, 2)],
    [(3, 1), (4, 3, 2)],
])
def test_add_diff_shape_reverse(shapes):
    run(shapes, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (3)],
    [(2, 3), (1)],
    [(4, 3, 2), (2)]
])
def add_with_scalar(shapes):
    run(shapes, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", [
    [(3), (2, 3)],
    [(1), (2, 3)],
    [(2), (4, 3, 2)]
])
def add_with_scalar_reversed(shapes):
    run(shapes, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (2, 3)],
    [(100, 100), (100, 100)],
    [(78, 91), (78, 91)],
    [(4, 3, 2), (4, 3, 2)],
    [(87, 3, 10), (87, 3, 10)],
])
def test_sub_same_shape(shapes):
    run(shapes, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (1, 3)],
    [(2, 3), (2, 1)],
    [(4, 3, 2), (3, 2)],
    [(4, 3, 2), (1, 2)],
    [(4, 3, 2), (3, 1)],
])
def test_sub_diff_shape(shapes):
    run(shapes, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", [
    [(1, 3), (2, 3)],
    [(2, 1), (2, 3)],
    [(3, 2), (4, 3, 2)],
    [(1, 2), (4, 3, 2)],
    [(3, 1), (4, 3, 2)],
])
def test_sub_diff_shape_reverse(shapes):
    run(shapes, lambda x, y: x-y)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (3)],
    [(2, 3), (1)],
    [(4, 3, 2), (2)]
])
def sub_with_scalar(shapes):
    run(shapes, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", [
    [(3), (2, 3)],
    [(1), (2, 3)],
    [(2), (4, 3, 2)]
])
def sub_with_scalar_reversed(shapes):
    run(shapes, lambda x, y: x+y)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (3, 2)],
    [(100, 100), (100, 100)],
    [(78, 91), (91, 10)],
])
def test_matmul(shapes):
    run(shapes, lambda x, y: x@y)

@pytest.mark.parametrize("shapes", [
    [(2, 3), (2, 3)],
    [(78, 91), (66, 91)],
])
def test_transpose_diff_tensors(shapes):
    run(shapes, lambda x, y: x@y.T)
    
@pytest.mark.parametrize("shapes", [
    [(66, 91), (66, 78)],
])
def test_transpose_diff_tensors_reverse(shapes):
    run(shapes, lambda x, y: x.T@y)

@pytest.mark.parametrize("shapes", [
    [(2, 3)],
    [(78, 91)],
])
def test_transpose_same_tensors(shapes):
    run(shapes, lambda x: x@x.T)
