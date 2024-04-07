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
                subprocess.check_output(args=['clang', '-O2', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', str(output_file.name)], input=prg.encode('utf-8'))
                add_dll = ctypes.CDLL(temp_file)

        add_dll.add.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        add_dll.add.restype = ctypes.POINTER(ctypes.c_float) 
        return Buffer(add_dll.add(x._data, y._data, x._len), temp_file)
