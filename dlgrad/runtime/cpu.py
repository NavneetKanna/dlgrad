"""
This should contain all ops to be performed on the cpu

"""
from __future__ import annotations
from dlgrad.c_code import C
from dlgrad.dtype import dtypes
import subprocess
import ctypes
import tempfile
from dlgrad.helpers import get_temp_loc, check_temp_file_exists, BroadcastHelper
from dlgrad.buffer import Buffer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dlgrad.tensor import Tensor


class CPU:
    @staticmethod
    def _add_axis_helper(x: Tensor, y: Tensor, dtype: dtypes, axis: int = None) -> Buffer:
        if not isinstance(x.data, Buffer): 
            return x.data + y.data

        c_dtype = dtypes.get_c_dtype(dtype) 
        name = f"cpu_{c_dtype}_add"
        temp_file = check_temp_file_exists(starts_with=name) 

        add_dll = None 
        data = None

        if temp_file:
            add_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}")
        else:
            if axis == 0:
                prg = C._add_axis0(c_dtype, out_len=BroadcastHelper.out_len) 
            elif axis == 1:
                prg = C._add_axis1(c_dtype, out_len=BroadcastHelper.out_len) 

            with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix=name) as output_file: 
                temp_file = str(output_file.name)
                subprocess.check_output(args=['clang', '-O2', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', temp_file], input=prg.encode('utf-8'))
                add_dll = ctypes.CDLL(temp_file)

        if axis == 0:
            add_dll.add_with_broadcasting.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            add_dll.add_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float) 
            # TODO: assuming y is getting broadcasted, maybe pass from dispatch ?
            data = add_dll.add_with_broadcasting(x.data._buffer, y.data._buffer, x.numel, y.numel, x.shape[1])
        elif axis == 1:
            add_dll.add_with_broadcasting.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
            add_dll.add_with_broadcasting.restype = ctypes.POINTER(ctypes.c_float) 
            data = add_dll.add_with_broadcasting(x.data._buffer, y.data._buffer, x.numel, y.numel)

        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)

    @staticmethod
    def _sum_axis_helper(x: Tensor, dtype: dtypes, axis: int = None) -> Buffer:
        # if not isinstance(x.data, Buffer): 
        #     return x.data + y.data

        c_dtype = dtypes.get_c_dtype(dtype) 
        name = f"cpu_{c_dtype}_sum"
        temp_file = check_temp_file_exists(starts_with=name) 

        sum_dll = None 
        data = None

        if temp_file:
            sum_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}")
        else:
            if axis == 0:
                prg = C._sum_axis0(c_dtype) 
            elif axis == 1:
                prg = C._sum_axis1(c_dtype) 

            with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix=name) as output_file: 
                temp_file = str(output_file.name)
                subprocess.check_output(args=['clang', '-O2', '-march=native', '-fPIC', '-x', 'c', '-', '-shared', '-o', temp_file], input=prg.encode('utf-8'))
                sum_dll = ctypes.CDLL(temp_file)

        if not axis:
            sum_dll.sum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            sum_dll.sum.restype = ctypes.POINTER(ctypes.c_int) 
            # TODO: assuming y is getting broadcasted, maybe pass from dispatch ?
            data = sum_dll.sum(x.data._buffer, x.numel)

        if axis == 0:
            sum_dll.sum_axis0.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            sum_dll.sum_axis0.restype = ctypes.POINTER(ctypes.c_float) 
            # TODO: assuming y is getting broadcasted, maybe pass from dispatch ?
            data = sum_dll.sum_axis0(x.data._buffer, x.numel, x.shape[0], x.shape[1])
        elif axis == 1:
            sum_dll.sum_axis1.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            sum_dll.sum_axis1.restype = ctypes.POINTER(ctypes.c_float) 
            data = sum_dll.sum_axis1(x.data._buffer, x.numel, x.shape[0], x.shape[1])

        if data is None:
            # TODO: create a new error
            print("Error: could not allocate memory")

        return Buffer(data, temp_file)
    
    @staticmethod
    def add_axis0(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        return CPU._add_axis_helper(x, y, dtype, axis=0)

    @staticmethod
    def _add_axis1(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        return CPU._add_axis_helper(x, y, dtype, axis=1)

    @staticmethod
    def sum_axis0(x: Tensor, dtype: dtypes) -> Buffer:
        return CPU._add_axis_helper(x, dtype, axis=0)

    @staticmethod
    def _sum_axis1(x: Tensor, dtype: dtypes) -> Buffer:
        return CPU._add_axis_helper(x, dtype, axis=1)

    @staticmethod
    def sum(x: Tensor, dtype: dtypes) -> Buffer:
        return CPU._add_axis_helper(x, dtype, axis=None)

    @staticmethod
    def matmul(x: Tensor, y: Tensor, dtype: dtypes) -> Buffer:
        if not isinstance(x.data, Buffer):
            pass
        else:
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
            data = matmul_dll.matmul(x.data._buffer, y.data._buffer, x.shape[0], x.shape[1], y.shape[1])
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
                transpose_dll = ctypes.CDLL(f"{get_temp_loc()}/{temp_file}")
            else:
                prg = C._transpose(c_dtype) 
                with tempfile.NamedTemporaryFile(delete=False, dir=get_temp_loc(), prefix=name) as output_file: 
                    temp_file = str(output_file.name)
                    subprocess.check_output(args=['clang', '-O2', '-march=native', '-x', 'c', '-', '-shared', '-o', temp_file], input=prg.encode('utf-8'))
                    transpose_dll = ctypes.CDLL(temp_file)

            transpose_dll.transpose.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
            transpose_dll.transpose.restype = ctypes.POINTER(ctypes.c_float) 
            data = transpose_dll.transpose(x.data._buffer, x.shape[0], x.shape[1])
            if data is None:
                # TODO: create a new error
                print("Error: could not allocate memory")
            return Buffer(data, temp_file)