# ruff: noqa

import functools
import os
import hashlib
import math
import pathlib
import struct
import sysconfig
from functools import cache

import _allocate  # type: ignore
import Metal
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.codegen import metal_kernel
from dlgrad.runtime.cpu import CPU
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, DType, Scalar
from dlgrad.helpers import BinaryOps, UnaryOps, cal_sum_max_out_shape, prod_, CACHE_DIR, calculate_stride


device = Metal.MTLCreateSystemDefaultDevice()
commandQueue = device.newCommandQueue()

class MetalGPU:
    """
    Main GPU runtime class for apple silicon gpus which handles the logic of using metal.

    This class uses PyObjC to interact with metal code.
    """
    ffi = FFI()

    @staticmethod
    def malloc(num: int, size: int = struct.calcsize('f')) -> CDataPtr:
        ptr = MetalGPU.ffi.gc(_allocate.lib.uninitialized_memory(num*size), _allocate.lib.free_ptr)
        if ptr == MetalGPU.ffi.NULL:
            raise MemoryError(f"Failed to allocate {num * size} bytes of memory")
        return ptr

    @staticmethod
    def calloc(num: int, size: int = struct.calcsize('f')) -> CDataPtr:
        ptr = MetalGPU.ffi.gc(_allocate.lib.initialized_memory(num, size), _allocate.lib.free_ptr)
        if ptr == MetalGPU.ffi.NULL:
            raise MemoryError(f"Failed to allocate {num * size} bytes of memory")
        return ptr

    @staticmethod
    @cache
    def build_1d_pipeline(src: str, func_name: str, numel: int):
        options = Metal.MTLCompileOptions.alloc().init()
        lib, _ = device.newLibraryWithSource_options_error_(src, options, None)
        fn = lib.newFunctionWithName_(func_name)
        pso = device.newComputePipelineStateWithFunction_error_(fn, None)[0]

        w = pso.maxTotalThreadsPerThreadgroup()
        threadsPerGrid = Metal.MTLSizeMake(numel, 1, 1)
        threadsPerThreadgroup = Metal.MTLSizeMake(min(w, numel), 1, 1)

        return pso, threadsPerGrid, threadsPerThreadgroup
    
    @staticmethod
    @cache
    def build_2d_pipeline(src: str, func_name: str, w: int, h: int):
        options = Metal.MTLCompileOptions.alloc().init()
        lib, _ = device.newLibraryWithSource_options_error_(src, options, None)
        # print(_)
        fn = lib.newFunctionWithName_(func_name)
        pso = device.newComputePipelineStateWithFunction_error_(fn, None)[0]

        threadsPerGrid = Metal.MTLSizeMake(w, h, 1)
        tw = pso.threadExecutionWidth() if w > pso.threadExecutionWidth() else w
        th = (pso.maxTotalThreadsPerThreadgroup() / tw) if h > (pso.maxTotalThreadsPerThreadgroup() / tw) else h
        threadsPerThreadgroup = Metal.MTLSizeMake(tw, th, 1)

        return pso, threadsPerGrid, threadsPerThreadgroup

    @staticmethod
    def _run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup) -> None:  # noqa: ANN001
        computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadgroup)
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

    @staticmethod
    def _binary_op(x: Buffer, y: Buffer, op: str) -> CDataPtr:  # noqa: ANN001
        out_ptr = MetalGPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(y.ptr, y.nbytes), y.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        
        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()
        
        src = metal_kernel.generate_binary_op_kernel(x.shape, x.stride, y.shape, y.stride, op=op)
        # print(src)
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_1d_pipeline(src, "binary_op", x.numel)
        
        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 2)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)
        
        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.METAL)
    def add(x: Buffer, y: Buffer) -> CDataPtr:
        return MetalGPU._binary_op(x, y, "add")

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.METAL)
    def sub(x: Buffer, y: Buffer) -> CDataPtr:
        return MetalGPU._binary_op(x, y, "sub")

    @staticmethod
    @dispatcher.register(BinaryOps.MUL, Device.METAL)
    def mul(x: Buffer, y: Buffer) -> CDataPtr:
        return MetalGPU._binary_op(x, y, "mul")

    @staticmethod
    @dispatcher.register(BinaryOps.DIV, Device.METAL)
    def div(x: Buffer, y: Buffer) -> CDataPtr:
        return MetalGPU._binary_op(x, y, "div")

    @staticmethod
    @dispatcher.register(BinaryOps.MATMUL, Device.METAL)
    def matmul(x: Buffer, y: Buffer):
        out_ptr = CPU.init_with_scalar(num=x.shape[0]*y.shape[1], scalar=0)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(y.ptr, y.nbytes), y.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()
        
        src = metal_kernel.matmul(x.shape, y.shape)
        # print(x.shape, y.shape)
        # print(src)
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "matmul", w=x.shape[0], h=y.shape[1])
        
        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 2)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.MAX, Device.METAL)
    def max(x: Buffer, dim: int, backward: bool = False, out: Buffer = None):
        out_shape = cal_sum_max_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape)
        num = prod_(out_shape)
        out_ptr = CPU.init_with_scalar(num=num, scalar=0)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()
        
        if x.ndim == 4:
            if dim == 0:
                src = metal_kernel.max_4d(x.shape, x.stride, dim=dim)
            elif dim == 1:
                src = metal_kernel.max_4d(x.shape, x.stride, dim=dim)

        # print(src)
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "max", w=out_shape[-1], h=prod_(out_shape[:-1]))
        # print("threadsPerGrid", threadsPerGrid)
        # print("threadsPerThreadgroup", threadsPerThreadgroup)
        # exit()
        
        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        return out_ptr