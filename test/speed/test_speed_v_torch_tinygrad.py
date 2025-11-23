import torch
import numpy as np
import time
from dlgrad import Tensor
import tinygrad
from dlgrad.helpers import get_color
import platform
import shutil
import os

if shutil.which('clang'):
   os.environ["CC"] = "clang"
elif shutil.which('gcc'):
    os.environ["CC"] = "gcc"

ITERATIONS = 8

def benchmark(func, data, device: str, sync_tiny: bool = False, sync_torch: bool = False) -> float:
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        result = func(*data)
        if sync_tiny:
            result = result.realize()
            tinygrad.Device[device].synchronize()
        if sync_torch:
            if platform.system() == 'Darwin':
                torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    return np.min(times)

def convert_time(seconds: float) -> tuple[float, str]:
    return round(seconds * 1e3, 2), "ms"

def to_torch_device(device):
    return "mps" if device == "metal" else device

def color_ratio(lib_time: float, dlgrad_time: float):
    if dlgrad_time == lib_time:
        return f"{get_color('end')}1.0", "equal"
    elif dlgrad_time < lib_time:  # dlgrad is faster
        ratio = lib_time / dlgrad_time
        return f"{get_color('green')}{ratio:.1f}{get_color('end')}", "faster"
    else:
        ratio = dlgrad_time / lib_time
        return f"{get_color('red')}{ratio:.1f}{get_color('end')}", "slower"

def run_benchmark(shapes: tuple, func, op_name: str, nargs: int, device: str):
    np_data = [np.random.uniform(size=shapes).astype(np.float32) for _ in range(nargs)]
    dlgrad_data = [Tensor(data, device=device) for data in np_data]
    tinygrad_data = [tinygrad.Tensor(data, device=device).realize() for data in np_data]
    torch_data = [torch.tensor(data, device=to_torch_device(device)) for data in np_data]

    dlgrad_time = benchmark(func, dlgrad_data, device=device)
    torch_time = benchmark(func, torch_data, sync_torch=True, device=device)
    tinygrad_time = benchmark(func, tinygrad_data, sync_tiny=True, device=device)

    torch_ratio, torch_desc = color_ratio(torch_time, dlgrad_time)
    tinygrad_ratio, tinygrad_desc = color_ratio(tinygrad_time, dlgrad_time)

    dlgrad_time, dlgrad_unit = convert_time(dlgrad_time)
    torch_time, torch_unit = convert_time(torch_time)
    tinygrad_time, tinygrad_unit = convert_time(tinygrad_time)

    print(
        f"{op_name:^12} | "
        f"{f'{shapes[0]}x{shapes[1]}':^12} | "
        f"{f'dlgrad: {dlgrad_time:.2f}{dlgrad_unit} Torch: {torch_time:.2f}{torch_unit} Tinygrad: {tinygrad_time:.2f}{tinygrad_unit}':^64} | "
        f"{f'vs Torch: {torch_ratio:15} ({torch_desc:})':^20} | "
        f"{f'vs Tinygrad: {tinygrad_ratio:15} ({tinygrad_desc})':^16} |"
    )

    np.testing.assert_allclose(
        func(*dlgrad_data).numpy(),
        func(*torch_data).cpu().numpy(),
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
    header = " CPU Benchmarks "
    print(f"{'=' * ((155 - len(header)) // 2)}{header}{'=' * ((155 - len(header)) // 2)}")
    print(f"{'Operation':^12} | {'Shape':^12} | {'Times':^64} | {'vs Torch':^25} | {'vs Tinygrad':^28} |")
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
        # (lambda x, y: x@y, "matmul", 2),
    ]

    print("-" * 155)
    for shape in shapes:
        for func, name, nargs in operations:
            run_benchmark(shape, func, name, nargs, "cpu")
    print("-" * 155)
    print()
    if platform.system() == 'Darwin':
        header = " Metal Benchmarks "
        print(f"{'=' * ((155 - len(header)) // 2)}{header}{'=' * ((155 - len(header)) // 2)}")
        print(f"{'Operation':^12} | {'Shape':^12} | {'Times':^64} | {'vs Torch':^25} | {'vs Tinygrad':^28}")
        shapes = [
            (64, 64),
            (4096, 4096)
        ]
        operations = [
            (lambda x, y: x + y, "add", 2),
            (lambda x, y: x - y, "sub", 2),
            (lambda x, y: x / y, "div", 2),
            (lambda x, y: x * y, "mul", 2),
            (lambda x: x.relu(), "relu", 1),
            (lambda x: x.exp(), "exp", 1),
            (lambda x: x.log(), "log", 1),
            (lambda x: x.sqrt(), "sqrt", 1),
            (lambda x: x**2, "pow", 1),
            (lambda x: x.max(), "max", 1),
            (lambda x: x.sum(), "sum", 1),
            (lambda x, y: x@y, "matmul", 2),
        ]

        print("-" * 155)
        for shape in shapes:
            for func, name, nargs in operations:
                run_benchmark(shape, func, name, nargs, "metal")
        print("-" * 155)

if __name__ == '__main__':
    test_all_operations()
