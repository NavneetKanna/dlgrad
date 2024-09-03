from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING, Optional

from dlgrad.buffer import Buffer
from dlgrad.c_code import C
from dlgrad.dtype import dtypes
from dlgrad.helpers import (BinaryOps, BufferOps, UnaryOps,
                            get_shared_lib_name, get_temp_loc, calculate_add_axis, calculate_numel, get_broadcast_shape, flatten)

if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


# TODO: is it dll or sha_lib ?
# TODO: Compile with len ? instead of giving it dynamically ?
# TODO: Am i saving to dict ?
class CPU:
    dlls: dict[ctypes.CDLL] = {}
    
    @staticmethod
    def _from_list(x: list) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtypes.from_py(type(x[0])), map_ctype=True)
        x = flatten(x)
        data = (c_dtype * len(x))(*x)

        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data)

    @staticmethod
    def _add(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtype)
        axis = -2 if x.numel == 1 or y.numel == 1 else calculate_add_axis(x.shape, y.shape)
        if axis is None:
            # TODO: Raise error
            print(f"add not compatiable with shapes {x.shape} / {y.shape}")
        name = get_shared_lib_name(f"add_axis{axis}", c_dtype, x.device.name)
        out_len = calculate_numel(get_broadcast_shape(x, y))

        if axis == 0:
            prg = C.add_axis0(c_dtype, out_len)
            add_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            add_dll.add_with_broadcasting.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            add_dll.add_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float)
            data = add_dll.add_with_broadcasting(x.data.buffer, y.data.buffer, x.numel, y.numel, x.shape[1])
        elif axis == 1:
            prg = C.add_axis1(c_dtype, out_len)
            add_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            add_dll.add_with_broadcasting.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int
            ]
            add_dll.add_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float)
            data = add_dll.add_with_broadcasting(x.data.buffer, y.data.buffer, x.numel, y.numel)
        elif axis == -1:
            prg = C.add_m1(c_dtype, out_len)
            add_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            add_dll.add_m1.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            add_dll.add_m1.restype = ctypes.POINTER(ctypes.c_float)
            data = add_dll.add_m1(x.data.buffer, y.data.buffer)
        else:
            prg = C.add_m2(c_dtype, out_len)
            add_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            add_dll.add_m2.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            add_dll.add_m2.restype = ctypes.POINTER(ctypes.c_float)
            data = add_dll.add_m2(x.data.buffer, y.data.buffer)

        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _sub(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtype)
        axis = -2 if x.numel == 1 or y.numel == 1 else calculate_add_axis(x.shape, y.shape)
        if axis is None:
            # TODO: Raise error
            print(f"subtraction not compatiable with shapes {x.shape} / {y.shape}")
        name = get_shared_lib_name(f"sub_axis{axis}", c_dtype, x.device.name)
        out_len = calculate_numel(get_broadcast_shape(x, y))

        if axis == 0:
            prg = C.sub_axis0(c_dtype, out_len)
            sub_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            sub_dll.sub_with_broadcasting.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            sub_dll.sub_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float)
            data = sub_dll.sub_with_broadcasting(x.data.buffer, y.data.buffer, x.numel, y.numel, x.shape[1])
        elif axis == 1:
            prg = C.sub_axis1(c_dtype, out_len)
            sub_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            sub_dll.sub_with_broadcasting.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int
            ]
            sub_dll.sub_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float)
            data = sub_dll.sub_with_broadcasting(x.data.buffer, y.data.buffer, x.numel, y.numel)
        elif axis == -1:
            prg = C.sub_m1(c_dtype, out_len)
            sub_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            sub_dll.sub_m1.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            sub_dll.sub_m1.restype = ctypes.POINTER(ctypes.c_float)
            data = sub_dll.sub_m1(x.data.buffer, y.data.buffer)
        else:
            prg = C.sub_m2(c_dtype, out_len)
            sub_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            sub_dll.sub_m2.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            sub_dll.sub_m2.restype = ctypes.POINTER(ctypes.c_float)
            data = sub_dll.sub_m2(x.data.buffer, y.data.buffer)

        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _div(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtype)
        axis = -2 if x.numel == 1 or y.numel == 1 else calculate_add_axis(x.shape, y.shape)
        if axis is None:
            # TODO: Raise error
            print(f"div not compatiable with shapes {x.shape} / {y.shape}")
        name = get_shared_lib_name(f"div_axis{axis}", c_dtype, x.device.name)
        out_len = calculate_numel(get_broadcast_shape(x, y))

        if axis == 0:
            prg = C.div_axis0(c_dtype, out_len)
            div_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            div_dll.div_with_broadcasting.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            div_dll.div_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float)
            data = div_dll.div_with_broadcasting(x.data.buffer, y.data.buffer, x.numel, y.numel, x.shape[1])
        elif axis == 1:
            prg = C.div_axis1(c_dtype, out_len)
            div_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            div_dll.div_with_broadcasting.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int
            ]
            div_dll.div_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float)
            data = div_dll.div_with_broadcasting(x.data.buffer, y.data.buffer, x.numel, y.numel)
        elif axis == -1:
            prg = C.div_m1(c_dtype, out_len)
            div_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            div_dll.div_m1.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            div_dll.div_m1.restype = ctypes.POINTER(ctypes.c_float)
            data = div_dll.div_m1(x.data.buffer, y.data.buffer)
        else:
            prg = C.div_m2(c_dtype, out_len)
            div_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            div_dll.div_m2.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            div_dll.div_m2.restype = ctypes.POINTER(ctypes.c_float)
            data = div_dll.div_m2(x.data.buffer, y.data.buffer)

        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _sum(x: Tensor, axis: int, dtype: dtypes) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtype)
        name = get_shared_lib_name(f"sum_axis{axis}", c_dtype, x.device.name)

        if axis == 0:
            prg = C.sum_axis0(c_dtype)
            sum_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            sum_dll.sum_axis0.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            sum_dll.sum_axis0.restype = ctypes.POINTER(ctypes.c_float)
            # TODO: assuming y is getting broadcasted, maybe pass from dispatch ?
            data = sum_dll.sum_axis0(x.data.buffer, x.numel, x.shape[0], x.shape[1])
        elif axis == 1:
            prg = C.sum_axis1(c_dtype)
            sum_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            sum_dll.sum_axis1.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            sum_dll.sum_axis1.restype = ctypes.POINTER(ctypes.c_float)
            data = sum_dll.sum_axis1(x.data.buffer, x.numel, x.shape[0], x.shape[1])
        else:
            prg = C.sum(c_dtype)
            sum_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))
            sum_dll.sum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            sum_dll.sum.restype = ctypes.POINTER(ctypes.c_float)
            data = sum_dll.sum(x.data.buffer, x.numel)

        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _matmul(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        if not isinstance(x.data, Buffer):
            pass
            
        c_dtype = dtypes.get_c_dtype(dtype)
        prg = C.matmul(c_dtype)
        name = get_shared_lib_name("matmul", c_dtype, x.device.name)
        matmul_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))

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
    def _transpose(x: Tensor, dtype: dtypes):
        if not isinstance(x.data, Buffer):
            pass

        c_dtype = dtypes.get_c_dtype(dtype)
        prg = C.transpose(c_dtype)
        name = get_shared_lib_name("transpose", c_dtype, x.device.name)
        transpose_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))

        transpose_dll.transpose.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        transpose_dll.transpose.restype = ctypes.POINTER(ctypes.c_float)
        data = transpose_dll.transpose(x.data.buffer, x.shape[0], x.shape[1])
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _uniform(length: int, low=0.0, high=1.0) -> Buffer:
        prg = C.random_buffer()
        name = get_shared_lib_name("uniform")
        rand_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))

        rand_dll.create_rand_buffer.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_float)
        rand_dll.create_rand_buffer.restype = ctypes.POINTER(ctypes.c_float)
        data = rand_dll.create_rand_buffer(length, low, high)
        print(data, type(data))
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _ones(length: int) -> Buffer:
        prg = C.ones_buffer()
        name = get_shared_lib_name("ones")
        ones_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))

        ones_dll.create_ones_buffer.argtypes = (ctypes.c_int,)
        ones_dll.create_ones_buffer.restype = ctypes.POINTER(ctypes.c_float)
        data = ones_dll.create_ones_buffer(length)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _relu(x: Tensor) -> Buffer:
        c_dtype = dtypes.get_c_dtype(x.dtype)
        prg = C.relu(c_dtype)
        name = get_shared_lib_name("relu")
        relu_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))

        relu_dll.relu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int)
        relu_dll.relu.restype = ctypes.POINTER(ctypes.c_float)
        data = relu_dll.relu(x.data.buffer, x.numel)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _max(x: Tensor) -> Buffer:
        c_dtype = dtypes.get_c_dtype(x.dtype)
        prg = C.max(c_dtype)
        name = get_shared_lib_name("max")
        max_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))

        max_dll.max.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int)
        max_dll.max.restype = ctypes.POINTER(ctypes.c_float)
        data = max_dll.max(x.data.buffer, x.numel)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _exp(x: Tensor) -> Buffer:
        c_dtype = dtypes.get_c_dtype(x.dtype)
        prg = C.exp(c_dtype)
        name = get_shared_lib_name("exp")
        exp_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))

        exp_dll.expp.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int)
        exp_dll.expp.restype = ctypes.POINTER(ctypes.c_float)
        data = exp_dll.expp(x.data.buffer, x.numel)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)
    
    @staticmethod
    def _log(x: Tensor) -> Buffer:
        c_dtype = dtypes.get_c_dtype(x.dtype)
        prg = C.log(c_dtype)
        name = get_shared_lib_name("log")
        log_dll, temp_file = CPU.dlls.get(name, CPU._compile_clang(name, prg))

        log_dll.logg.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int)
        log_dll.logg.restype = ctypes.POINTER(ctypes.c_float)
        data = log_dll.logg(x.data.buffer, x.numel)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
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

    # TODO: is or == ?
    @staticmethod
    def interface(op, x: Optional[Tensor] = None, y: Optional[Tensor] = None, **kwargs) -> Buffer:
        axis = kwargs.get("axis", None)
        func = kwargs.get("func", None)

        if isinstance(op, BinaryOps):
            if op == BinaryOps.ADD:
                return CPU._add(x, y, x.dtype)
            if op == BinaryOps.SUB:
                return CPU._sub(x, y, x.dtype)
            if op == BinaryOps.DIV:
                return CPU._div(x, y, x.dtype)
            if op == BinaryOps.MATMUL:
                return CPU._matmul(x, y, x.dtype)

        elif isinstance(op, UnaryOps):
            if op == UnaryOps.SUM:
                return CPU._sum(x, axis, x.dtype)
            if op == UnaryOps.TRANSPOSE:
                return CPU._transpose(x, x.dtype)
            if op == UnaryOps.MAX:
                if func == "max":
                    return CPU._max(x)
                elif func == "relu":
                    return CPU._relu(x)
            if op == UnaryOps.EXP:
                return CPU._exp(x)
            if op == UnaryOps.LOG:
                return CPU._log(x)

        elif isinstance(op, BufferOps):
            if op == BufferOps.UNIFORM:
                return CPU._uniform(kwargs["out_len"], kwargs["low"], kwargs["high"])
            if op == BufferOps.ONES:
                return CPU._ones(kwargs["out_len"])
            if op == BufferOps.CUSTOM:
                if func == "from_list":
                    return CPU._from_list(kwargs["li"])


        raise ValueError(f"Unsupported operation: {op}")
