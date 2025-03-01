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
import tinygrad
import math


ITER = 8
def _run(func, data, tiny = False):
    t = []
    for _ in range(ITER):
        s = time.perf_counter()
        func(*data) if not tiny else func(*data).realize()
        e = time.perf_counter()
        t.append(e-s)
    
    return np.min(t)

def convert_time(time_in_s: float):
    nzeros = abs(int(math.log10(abs(time_in_s))))

    if 0 <= nzeros <= 3: # ms
        return round(time_in_s * 1e3, 2), "ms"
    elif 3 < nzeros <= 6: # us
        return round(time_in_s * 1e6, 2), "us"
    elif 6 < nzeros <= 9: # ns
        return round(time_in_s * 1e9, 2), "ns"

# TODO: Colorize
def run(shapes: list[tuple], func, op_name: str, nargs: int):
    np_data = [np.random.uniform(size=shapes[0]).astype(np.float32) for _ in range(nargs)]
    dlgrad_data = [Tensor(data) for data in np_data]
    tinygrad_data = [tinygrad.Tensor(data, device="cpu") for data in np_data]
    torch_data = [torch.tensor(data, device="cpu") for data in np_data]

    dlgrad_time = _run(func=func, data=dlgrad_data)
    torch_time = _run(func=func, data=torch_data)
    tinygrad_time = _run(func=func, data=tinygrad_data, tiny=True)

    torch_ratio = torch_time / dlgrad_time if dlgrad_time < torch_time else dlgrad_time / torch_time
    torch_desc = "faster" if dlgrad_time < torch_time else "slower"

    tinygrad_ratio = tinygrad_time / dlgrad_time if dlgrad_time < tinygrad_time else dlgrad_time / tinygrad_time
    tinygrad_desc = "faster" if dlgrad_time < tinygrad_time else "slower"
    
    dlgrad_time, dl_unit = convert_time(_run(func=func, data=dlgrad_data))
    torch_time, to_unit = convert_time(_run(func=func, data=torch_data))
    tinygrad_time, ti_unit = convert_time(_run(func=func, data=tinygrad_data, tiny=True))

    print(
        f"{op_name:<20} ({shapes[0][0]:5d}, {shapes[0][0]:5d}) "
        f"dlgrad: {dlgrad_time}{dl_unit}, "
        f"Torch: {torch_time}{to_unit} -> {torch_ratio:.2f} {torch_desc} "
        f"Tinygrad: {tinygrad_time}{ti_unit}  -> {tinygrad_ratio:.2f} {tinygrad_desc}"
    )

    np.testing.assert_allclose(
        func(*dlgrad_data).numpy(),
        func(*torch_data).numpy(),
        atol=1e-6,
        rtol=1e-3
    )
    np.testing.assert_allclose(
        func(*dlgrad_data).numpy(),
        func(*tinygrad_data).numpy(),
        atol=1e-6,
        rtol=1e-3
    )

shapes_list = [
    [(20, 30)],
    [(4096, 4096)]
]

operations = [
    (lambda x, y: x + y, "add", 2),
    (lambda x, y: x - y, "sub", 2),
    (lambda x, y: x / y, "div", 2),
    (lambda x, y: x * y, "mul", 2),
    (lambda x: x.relu(), "relu", 1),
    (lambda x: x**2, "pow", 1),
    (lambda x: x.sum(), "sum", 1),
    (lambda x: x.max(), "max", 1),
]

def test_all_ops():
    for j in shapes_list:
        for i in operations:
            run(j, i[0], i[1], i[2])
