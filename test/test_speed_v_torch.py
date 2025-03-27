import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import numpy as np
import time
from dlgrad import Tensor
import tinygrad
import math
from dlgrad.helpers import get_color

ITERATIONS = 8

def benchmark(func, data, use_tiny: bool = False) -> float:
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        result = func(*data)
        if use_tiny:
            result = result.realize()
        times.append(time.perf_counter() - start)
    return np.min(times)

def convert_time(seconds: float) -> tuple[float, str]:
    return round(seconds * 1e3, 2), "ms"
    # if seconds == 0:
    #     return 0.0, "s"
    # nzeros = abs(int(math.log10(abs(seconds))))
    # if 0 <= nzeros <= 3:      # milliseconds
    #     return round(seconds * 1e3, 2), "ms"
    # elif 3 < nzeros <= 6:     # microseconds
    #     return round(seconds * 1e6, 2), "µs"
    # elif 6 < nzeros <= 9:     # nanoseconds
    #     return round(seconds * 1e9, 2), "ns"
    # else:
    #     return round(seconds, 2), "s"

def color_ratio(lib_time: float, dlgrad_time: float):
    if dlgrad_time == lib_time:
        return f"{get_color('end')}1.0", "equal"
    elif dlgrad_time < lib_time:  # dlgrad is faster
        ratio = lib_time / dlgrad_time
        return f"{get_color('green')}{ratio:.1f}{get_color('end')}", "faster"
    else:  
        ratio = dlgrad_time / lib_time
        return f"{get_color('red')}{ratio:.1f}{get_color('end')}", "slower"
    
def run_benchmark(shapes: tuple, func, op_name: str, nargs: int):
    np_data = [np.random.uniform(size=shapes).astype(np.float32) for _ in range(nargs)]
    dlgrad_data = [Tensor(data) for data in np_data]
    tinygrad_data = [tinygrad.Tensor(data, device="cpu") for data in np_data]
    torch_data = [torch.tensor(data, device="cpu") for data in np_data]

    dlgrad_time = benchmark(func, dlgrad_data)
    torch_time = benchmark(func, torch_data)
    tinygrad_time = benchmark(func, tinygrad_data, use_tiny=True)

    torch_ratio, torch_desc = color_ratio(torch_time, dlgrad_time)
    tinygrad_ratio, tinygrad_desc = color_ratio(tinygrad_time, dlgrad_time)

    dlgrad_time, dlgrad_unit = convert_time(dlgrad_time)
    torch_time, torch_unit = convert_time(torch_time)
    tinygrad_time, tinygrad_unit = convert_time(tinygrad_time)

    print(
        f"{op_name:^12} | "                      
        f"{f'{shapes[0]}x{shapes[1]}':^12} | "    
        f"{f'dlgrad: {dlgrad_time:.2f}{dlgrad_unit} Torch: {torch_time:.2f}{torch_unit} Tinygrad: {tinygrad_time:.2f}{tinygrad_unit}':^64} | "  # Times: 64 chars
        f"{f'vs Torch: {torch_ratio} ({torch_desc})':^16} | " 
        f"{f'vs Tinygrad: {tinygrad_ratio} ({tinygrad_desc})':^16}" 
    )
    print()

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

def test_all_operations() -> None:
    shapes = [
        (20, 20),
        (4096, 4096)
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
        (lambda x: x.exp(), "exp", 1),
        (lambda x: x.log(), "log", 1),
        (lambda x: x.sqrt(), "sqrt", 1),
    ] 
    # print(
    #     f"{'Operation':<10} | {'Shape':>11} | {'Times':>56} | {'vs Torch':>22} | {'vs Tinygrad':>15}"
    # )
    print("-" * 150)
    for shape in shapes:
        for func, name, nargs in operations:
            run_benchmark(shape, func, name, nargs)
    print("-" * 150)

if __name__ == '__main__':
    test_all_operations()
