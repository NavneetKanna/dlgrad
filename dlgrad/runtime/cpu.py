from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING 

from dlgrad.buffer import Buffer
from dlgrad.c_code import C
from dlgrad.dtype import dtypes
from dlgrad.helpers import BroadcastHelper, get_shared_lib_name, get_temp_loc

if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


def _compile_clang(name: str, prg: str) -> tuple[ctypes.CDLL, str]:
    with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix=name) as output_file:
        temp_file = str(output_file.name)
        subprocess.check_output(
            args=[
                "clang", "-O3", "-march=native", "-ffast-math", "-funroll-loops", "-fPIC",
                "-x", "c", "-", "-shared", "-o", temp_file
            ],
            input=prg.encode("utf-8"),
        )
        dll = ctypes.CDLL(temp_file, mode=os.RTLD_LAZY)
        CPU.dlls[name] = dll, temp_file

    return dll, temp_file


# TODO: is it dll or sha_lib ?
class CPU:
    dlls: dict[ctypes.CDLL] = {}
    
    @staticmethod
    def add(x: Tensor, y: Tensor, axis: int, dtype: dtypes) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtype)
        name = get_shared_lib_name(f"add_axis{axis}", c_dtype, x.device.name)

        if axis == 0:
            prg = C.add_axis0(c_dtype, out_len=BroadcastHelper.out_len)
            add_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))
            add_dll.add_with_broadcasting.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            add_dll.add_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float)
            data = add_dll.add_with_broadcasting(x.data.buffer, y.data.buffer, x.numel, y.numel, x.shape[1])
        elif axis == 1:
            prg = C.add_axis1(c_dtype, out_len=BroadcastHelper.out_len)
            add_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))
            add_dll.add_with_broadcasting.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int
            ]
            add_dll.add_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float)
            data = add_dll.add_with_broadcasting(x.data.buffer, y.data.buffer, x.numel, y.numel)
        else:
            prg = C.add(c_dtype, out_len=BroadcastHelper.out_len)
            add_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))
            add_dll.add.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            add_dll.add.restype = ctypes.POINTER(ctypes.c_float)
            data = add_dll.add(x.data.buffer, y.data.buffer)

        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)
    
    @staticmethod
    def sum_axis0(x: Tensor, dtype: dtypes) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtype)
        prg = C.sum_axis0(c_dtype)
        name = get_shared_lib_name("sum0", c_dtype, x.device.name)
        sum_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))

        sum_dll.sum_axis0.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        sum_dll.sum_axis0.restype = ctypes.POINTER(ctypes.c_float)
        # TODO: assuming y is getting broadcasted, maybe pass from dispatch ?
        data = sum_dll.sum_axis0(x.data.buffer, x.numel, x.shape[0], x.shape[1])
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)
    
    @staticmethod
    def sum_axis1(x: Tensor, dtype: dtypes) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtype)
        prg = C.sum_axis1(c_dtype)
        name = get_shared_lib_name("sum1", c_dtype, x.device.name)
        sum_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))

        sum_dll.sum_axis1.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        sum_dll.sum_axis1.restype = ctypes.POINTER(ctypes.c_float)
        data = sum_dll.sum_axis1(x.data.buffer, x.numel, x.shape[0], x.shape[1])
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def sum(x: Tensor, dtype: dtypes) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtype)
        prg = C.sum(c_dtype)
        name = get_shared_lib_name("sum", c_dtype, x.device.name)
        sum_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))

        sum_dll.sum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        sum_dll.sum.restype = ctypes.c_float
        data = sum_dll.sum(x.data.buffer, x.numel)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def matmul(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        if not isinstance(x.data, Buffer):
            pass
            
        c_dtype = dtypes.get_c_dtype(dtype)
        prg = C.matmul(c_dtype)
        name = get_shared_lib_name("matmul", c_dtype, x.device.name)
        matmul_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))

        matmul_dll.matmul.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        matmul_dll.matmul.restype = ctypes.POINTER(ctypes.c_float)
        data = matmul_dll.matmul(x.data.buffer, y.data.buffer, x.shape[0], x.shape[1], y.shape[1])
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def transpose(x: Tensor, dtype: dtypes):
        if not isinstance(x.data, Buffer):
            pass

        c_dtype = dtypes.get_c_dtype(dtype)
        prg = C.transpose(c_dtype)
        name = get_shared_lib_name("transpose", c_dtype, x.device.name)
        transpose_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))

        transpose_dll.transpose.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        transpose_dll.transpose.restype = ctypes.POINTER(ctypes.c_float)
        data = transpose_dll.transpose(x.data.buffer, x.shape[0], x.shape[1])
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def uniform(length: int, low=0.0, high=1.0) -> Buffer:
        prg = C.random_buffer()
        name = get_shared_lib_name("uniform")
        rand_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))

        rand_dll.create_rand_buffer.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_float)
        rand_dll.create_rand_buffer.restype = ctypes.POINTER(ctypes.c_float)
        data = rand_dll.create_rand_buffer(length, low, high)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def ones(length: int) -> Buffer:
        prg = C.ones_buffer()
        name = get_shared_lib_name("ones")
        ones_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))

        ones_dll.create_ones_buffer.argtypes = (ctypes.c_int,)
        ones_dll.create_ones_buffer.restype = ctypes.POINTER(ctypes.c_float)
        data = ones_dll.create_ones_buffer(length)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def relu(x: Tensor) -> Buffer:
        c_dtype = dtypes.get_c_dtype(x.dtype)
        prg = C.relu(c_dtype)
        name = get_shared_lib_name("relu")
        relu_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))

        relu_dll.relu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int)
        relu_dll.relu.restype = ctypes.POINTER(ctypes.c_float)
        data = relu_dll.relu(x.data.buffer, x.numel)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def exp(x: Tensor) -> Buffer:
        c_dtype = dtypes.get_c_dtype(x.dtype)
        prg = C.exp(c_dtype)
        name = get_shared_lib_name("exp")
        exp_dll, temp_file = CPU.dlls.get(name, _compile_clang(name, prg))

        exp_dll.exp.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int)
        exp_dll.exp.restype = ctypes.POINTER(ctypes.c_float)
        data = exp_dll.exp(x.data.buffer, x.numel)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)
    
    @staticmethod
    def interface():
        pass