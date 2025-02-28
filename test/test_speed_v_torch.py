import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import numpy as np
import pytest
import time
from dlgrad import Tensor


# TODO: Convert s to ms, us, ns
# TODO: Colorize
def run(shapes: list[tuple], func, op_name: str, nargs: int):
    np_data = [np.random.uniform(size=shapes[0]).astype(np.float32) for _ in range(nargs)]
    dlgrad_data = [Tensor(data) for data in np_data]
    torch_data = [torch.tensor(data, device="cpu") for data in np_data]

    np.testing.assert_allclose(
        func(*dlgrad_data).numpy(),
        func(*torch_data).numpy(),
        atol=1e-6,
        rtol=1e-3
    )

    for _ in range(10):
        func(*dlgrad_data)
        func(*torch_data)
        
    n_iter = 1000

    s = time.perf_counter()
    for _ in range(n_iter):
        func(*dlgrad_data)
    dl_time = (time.perf_counter() - s) / n_iter

    s = time.perf_counter()
    for _ in range(n_iter):
        func(*torch_data)
    torch_time = (time.perf_counter() - s) / n_iter

    if dl_time < torch_time:
        torch_ratio = torch_time / dl_time
        torch_desc = "faster"
    else:
        torch_ratio = dl_time / torch_time
        torch_desc = "slower"

    print(
        f"\n{op_name: <20} {shapes[0]} "
        f"Torch: {torch_time:.9f}s, dlgrad: {dl_time:.9f}s -> {torch_ratio:.2f}x {torch_desc}"
    )

shapes_list = [
    [(20, 20)]
]

@pytest.mark.parametrize("shapes", shapes_list)
def test_add(shapes):
    run(shapes, lambda x, y: x + y, "add", 2)

@pytest.mark.parametrize("shapes", shapes_list)
def test_sub(shapes):
    run(shapes, lambda x, y: x - y, "sub", 2)

@pytest.mark.parametrize("shapes", shapes_list)
def test_div(shapes):
    run(shapes, lambda x, y: x / y, "div", 2)

@pytest.mark.parametrize("shapes", shapes_list)
def test_mul(shapes):
    run(shapes, lambda x, y: x * y, "mul", 2)

@pytest.mark.parametrize("shapes", shapes_list)
def test_relu(shapes):
    run(shapes, lambda x: x.relu(), "relu", 1)
