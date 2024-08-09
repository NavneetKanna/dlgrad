from __future__ import annotations

import os
import ctypes
import subprocess
import tempfile
from typing import TYPE_CHECKING, Optional

from dlgrad.buffer import Buffer
from dlgrad.c_code import C
from dlgrad.dtype import dtypes
from dlgrad.helpers import BroadcastHelper, check_temp_file_exists, get_temp_loc

if TYPE_CHECKING:
    from dlgrad.tensor import Tensor



# TODO: Test diff flags with dlopen (https://manpages.debian.org/bookworm/manpages-dev/dlopen.3.en.html)
class CPU:
    @staticmethod
    def _add_axis_helper(
        x: Tensor, y: Tensor, dtype: dtypes, axis: Optional[int]
    ) -> Buffer:
        if not isinstance(x.data, Buffer):
            return x.data + y.data

        c_dtype = dtypes.get_c_dtype(dtype)
        name = f"cpu_{c_dtype}_add"
        temp_file = check_temp_file_exists(starts_with=name)

        add_dll = None
        data = None
        prg = None

        if axis == 0:
            prg = C.add_axis0(c_dtype, out_len=BroadcastHelper.out_len)
            name = f"cpu_{c_dtype}_add0"
            temp_file = check_temp_file_exists(starts_with=name)
        elif axis == 1:
            prg = C.add_axis1(c_dtype, out_len=BroadcastHelper.out_len)
            name = f"cpu_{c_dtype}_add1"
            temp_file = check_temp_file_exists(starts_with=name)
        elif axis == -1:
            prg = C.add(c_dtype, out_len=BroadcastHelper.out_len)
            name = f"cpu_{c_dtype}_add"
            temp_file = check_temp_file_exists(starts_with=name)

        if temp_file:
            add_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}", mode=os.RTLD_LAZY)
        else:
            with tempfile.NamedTemporaryFile(
                delete=False, dir=get_temp_loc(), prefix=name
            ) as output_file:
                temp_file = str(output_file.name)
                subprocess.check_output(
                    args=[
                        "clang",
                        "-O3",
                        "-march=native",
                        "-ffast-math",
                        "-funroll-loops",
                        "-fPIC",
                        "-x",
                        "c",
                        "-",
                        "-shared",
                        "-o",
                        temp_file,
                    ],
                    input=prg.encode("utf-8"),
                )
                add_dll = ctypes.CDLL(temp_file, mode=os.RTLD_LAZY)

        if axis == 0:
            add_dll.add_with_broadcasting.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            add_dll.add_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float)
            # TODO: assuming y is getting broadcasted, maybe pass from dispatch ?
            import time
            s = time.perf_counter()
            data = add_dll.add_with_broadcasting(
                x.data.buffer, y.data.buffer, x.numel, y.numel, x.shape[1]
            )
            e = time.perf_counter()
            print(f"actual {e-s:.4f}s")
        elif axis == 1:
            add_dll.add_with_broadcasting.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
            add_dll.add_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float)
            import time 
            s = time.perf_counter()
            data = add_dll.add_with_broadcasting(
                x.data.buffer, y.data.buffer, x.numel, y.numel
            )
            e = time.perf_counter()
            print(f"actual {e-s:.4f}s")
        elif axis == -1:
            add_dll.add.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            add_dll.add.restype = ctypes.POINTER(ctypes.c_float)
            import time
            s = time.perf_counter()
            data = add_dll.add(x.data.buffer, y.data.buffer)
            e = time.perf_counter()
            print(f"actual {e-s:.4f}s")

        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _sum_axis_helper(x: Tensor, dtype: dtypes, axis: Optional[int]) -> Buffer:
        # if not isinstance(x.data, Buffer):
        #     return x.data + y.data

        c_dtype = dtypes.get_c_dtype(dtype)

        sum_dll = None
        data = None
        prg = None
        temp_file = None
        name = None

        if axis == 0:
            prg = C.sum_axis0(c_dtype)
            name = f"cpu_{c_dtype}_sum0"
            temp_file = check_temp_file_exists(starts_with=name)
        elif axis == 1:
            prg = C.sum_axis1(c_dtype)
            name = f"cpu_{c_dtype}_sum1"
            temp_file = check_temp_file_exists(starts_with=name)
        elif axis == -1:
            prg = C.sum(c_dtype)
            name = f"cpu_{c_dtype}_sum"
            temp_file = check_temp_file_exists(starts_with=name)

        if temp_file:
            sum_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}", mode=os.RTLD_LAZY)
        else:
            with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix=name) as output_file:
                temp_file = str(output_file.name)
                subprocess.check_output(
                    args=[
                        "clang",
                        "-O2",
                        "-march=native",
                        "-ffast-math",
                        "-ftree-vectorize",
                        "-funroll-loops",
                        "-fPIC",
                        "-x",
                        "c",
                        "-",
                        "-shared",
                        "-o",
                        temp_file,
                    ],
                    input=prg.encode("utf-8"),
                )
                sum_dll = ctypes.CDLL(temp_file, mode=os.RTLD_LAZY)

        if axis == 0:
            sum_dll.sum_axis0.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            sum_dll.sum_axis0.restype = ctypes.POINTER(ctypes.c_float)
            # TODO: assuming y is getting broadcasted, maybe pass from dispatch ?
            data = sum_dll.sum_axis0(x.data.buffer, x.numel, x.shape[0], x.shape[1])
        elif axis == 1:
            sum_dll.sum_axis1.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            sum_dll.sum_axis1.restype = ctypes.POINTER(ctypes.c_float)
            data = sum_dll.sum_axis1(x.data.buffer, x.numel, x.shape[0], x.shape[1])
        elif axis == -1:
            sum_dll.sum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            sum_dll.sum.restype = ctypes.POINTER(ctypes.c_int)
            import time
            s = time.perf_counter()
            data = sum_dll.sum(x.data.buffer, x.numel)
            e = time.perf_counter()
            print(f"actual: {e-s:.4f}s")

        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def add(x: Tensor, y: Tensor, dtype: dtypes, axis: int) -> Buffer:
        return CPU._add_axis_helper(x, y, dtype, axis)

    @staticmethod
    def sum(x: Tensor, dtype: dtypes, axis: int) -> Buffer:
        return CPU._sum_axis_helper(x, dtype, axis)

    @staticmethod
    def matmul(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        if not isinstance(x.data, Buffer):
            pass
        else:
            c_dtype = dtypes.get_c_dtype(dtype)
            name = f"cpu_{c_dtype}_matmul"
            temp_file = check_temp_file_exists(starts_with=name)

            if temp_file:
                matmul_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}", mode=os.RTLD_LAZY)
            else:
                prg = C.matmul(c_dtype)
                with tempfile.NamedTemporaryFile(
                    delete=False, dir=get_temp_loc(), prefix=name
                ) as output_file:
                    temp_file = str(output_file.name)
                    subprocess.check_output(
                        args=[
                            "clang",
                            "-O2",
                            "-march=native",
                            "-x",
                            "c",
                            "-",
                            "-shared",
                            "-o",
                            temp_file,
                        ],
                        input=prg.encode("utf-8"),
                    )
                    matmul_dll = ctypes.CDLL(temp_file, mode=os.RTLD_LAZY)

            matmul_dll.matmul.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            matmul_dll.matmul.restype = ctypes.POINTER(ctypes.c_float)
            data = matmul_dll.matmul(
                x.data.buffer, y.data.buffer, x.shape[0], x.shape[1], y.shape[1]
            )
            if data is None:
                # TODO: create a new error
                print("Error: could not allocate memory")
            return Buffer(data, temp_file)

    @staticmethod
    def transpose(x: Tensor, dtype: dtypes):
        if not isinstance(x.data, Buffer):
            pass
        else:
            c_dtype = dtypes.get_c_dtype(dtype)
            name = f"cpu_{c_dtype}_transpose"
            temp_file = check_temp_file_exists(starts_with=name)

            if temp_file:
                transpose_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}", mode=os.RTLD_LAZY)
            else:
                prg = C.transpose(c_dtype)
                with tempfile.NamedTemporaryFile(
                    delete=False, dir=get_temp_loc(), prefix=name
                ) as output_file:
                    temp_file = str(output_file.name)
                    subprocess.check_output(
                        args=[
                            "clang",
                            "-O2",
                            "-march=native",
                            "-x",
                            "c",
                            "-",
                            "-shared",
                            "-o",
                            temp_file,
                        ],
                        input=prg.encode("utf-8"),
                    )
                    transpose_dll = ctypes.CDLL(temp_file, mode=os.RTLD_LAZY)

            transpose_dll.transpose.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
            transpose_dll.transpose.restype = ctypes.POINTER(ctypes.c_float)
            data = transpose_dll.transpose(x.data.buffer, x.shape[0], x.shape[1])
            if data is None:
                # TODO: create a new error
                print("Error: could not allocate memory")
            return Buffer(data, temp_file)

    @staticmethod
    def uniform(length: int, low=0.0, high=1.0) -> Buffer:
        temp_file = check_temp_file_exists(starts_with="rand_buffer")
        if temp_file:
            temp_file = f"{get_temp_loc()}/{temp_file}"
            rand_dll = ctypes.CDLL(temp_file)
        else:
            prg = C.random_buffer()
            with tempfile.NamedTemporaryFile(
                delete=False, dir=get_temp_loc(), prefix="rand_buffer"
            ) as output_file:
                temp_file = str(output_file.name)
                subprocess.check_output(
                    args=[
                        "clang",
                        "-O3",
                        "-march=native",
                        "-ffast-math",
                        "-fPIC",
                        "-x",
                        "c",
                        "-",
                        "-shared",
                        "-o",
                        temp_file,
                    ],
                    input=prg.encode("utf-8"),
                )
                rand_dll = ctypes.CDLL(temp_file)

        rand_dll.create_rand_buffer.argtypes = (
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
        )
        rand_dll.create_rand_buffer.restype = ctypes.POINTER(ctypes.c_float)
        data = rand_dll.create_rand_buffer(length, low, high)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")
        return Buffer(data, temp_file)

    @staticmethod
    def ones(length: int) -> Buffer:
        temp_file = check_temp_file_exists(starts_with="ones_buffer")
        if temp_file:
            temp_file = f"{get_temp_loc()}/{temp_file}"
            ones_dll = ctypes.CDLL(temp_file, mode=os.RTLD_LAZY)
        else:
            prg = C.ones_buffer()
            with tempfile.NamedTemporaryFile(
                delete=False, dir=get_temp_loc(), prefix="ones_buffer"
            ) as output_file:
                temp_file = str(output_file.name)
                subprocess.check_output(
                    args=[
                        "clang",
                        "-O2",
                        "-march=native",
                        "-fPIC",
                        "-x",
                        "c",
                        "-",
                        "-shared",
                        "-o",
                        temp_file,
                    ],
                    input=prg.encode("utf-8"),
                )
                ones_dll = ctypes.CDLL(temp_file, mode=os.RTLD_LAZY)

        ones_dll.create_ones_buffer.argtypes = (ctypes.c_int,)
        ones_dll.create_ones_buffer.restype = ctypes.POINTER(ctypes.c_float)
        data = ones_dll.create_ones_buffer(length)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")
        return Buffer(data, temp_file)
