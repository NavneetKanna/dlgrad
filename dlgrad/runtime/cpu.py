"""
This should contain all ops to be performed on the cpu

"""
from __future__ import annotations
from dlgrad.c_code import C
from dlgrad.dtype import dtypes
import subprocess
import ctypes
import tempfile
from dlgrad.helpers import get_temp_loc, check_temp_file_exists
from dlgrad.buffer import Buffer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class CPU:
    # TODO: should these be private ? 
    @staticmethod
    def add(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtype) 
        name = f"cpu_{c_dtype}_add"
        temp_file = check_temp_file_exists(starts_with=name) 

        if temp_file:
            add_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}")
        else:
            prg = C._add(c_dtype, out_len=x._len) 
            with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix=name) as output_file: 
                temp_file = str(output_file.name)
                subprocess.check_output(args=['clang', '-O2', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', temp_file], input=prg.encode('utf-8'))
                add_dll = ctypes.CDLL(temp_file)

        add_dll.add.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        add_dll.add.restype = ctypes.POINTER(ctypes.c_float) 
        data = add_dll.add(x._data, y._data)
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")
        return Buffer(data, temp_file)

    @staticmethod
    def matmul(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        c_dtype = dtypes.get_c_dtype(dtype) 
        name = f"cpu_{c_dtype}_matmul"
        temp_file = check_temp_file_exists(starts_with=name) 

        if temp_file:
            matmul_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}")
        else:
            prg = C._matmul(c_dtype) 
            with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix=name) as output_file: 
                temp_file = str(output_file.name)
                subprocess.check_output(args=['clang', '-O2', '-march=native', '-x', 'c', '-', '-shared', '-o', temp_file], input=prg.encode('utf-8'))
                matmul_dll = ctypes.CDLL(temp_file)

        matmul_dll.matmul.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        matmul_dll.matmul.restype = ctypes.POINTER(ctypes.c_float) 
        data = matmul_dll.matmul(x._data, y._data, x._shape[0], x._shape[1], y._shape[1])
        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")
        return Buffer(data, temp_file)

    @staticmethod
    def transpose(x: Tensor):
        pass