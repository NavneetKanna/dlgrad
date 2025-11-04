# ruff: noqa

import struct
from functools import cache

import _allocate  # type: ignore
import Metal
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.codegen import metal_kernel
from dlgrad.runtime.cpu import CPU
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, DType
from dlgrad.helpers import BinaryOps, UnaryOps, cal_sum_max_out_shape, prod_
from dlgrad.buffer import Buffer
from dlgrad.dtype import Scalar

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

        src = metal_kernel.arithmetic(x.shape, x.stride, y.shape, y.stride, op=op)
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
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.shape[0]*y.shape[1]*4), x.shape[0]*y.shape[1]*4, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        # src = metal_kernel.matmul_fast(x.shape, y.shape)
        src = metal_kernel.matmul(x.shape, y.shape)
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "matmul", w=y.shape[1], h=x.shape[0])
        # pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "matmul", w=32 * (y.shape[1]/8), h=32*(x.shape[0]/8))
        # threadgroupsPerGrid = Metal.MTLSizeMake(y.shape[1]/8, x.shape[0]/8, 1)
        # threadsPerThreadgroup = Metal.MTLSizeMake(32, 1, 1)

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 2)

        # computeEncoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroupsPerGrid, threadsPerThreadgroup)
        # computeEncoder.endEncoding()
        # commandBuffer.commit()
        # commandBuffer.waitUntilCompleted()

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    def reduce(x: Buffer, dim: int, op: str):
        if dim == -1:
            out_shape = cal_sum_max_out_shape(ndim=x.ndim, dim=len(x.shape) - 1, inp_shape=x.shape)
            num = prod_(out_shape)
            out_ptr = CPU.init_with_scalar(num=num, scalar=0)
        else:
            out_shape = cal_sum_max_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape)
            num = prod_(out_shape)
            out_ptr = CPU.init_with_scalar(num=num, scalar=0)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, num*4), num*4, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        if x.ndim == 4:
            src = metal_kernel.reduce_4d(x.shape, x.stride, dim=dim if dim != -1 else 3, op=op)
            if dim == 3 or dim == -1:
                pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, op, w=x.shape[-1], h=prod_(x.shape[:-1]))
                threadsPerGrid = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], prod_(x.shape[:-1]), 1)
                threadsPerThreadgroup = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], 1, 1)
            else:
                pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, op, w=out_shape[-1], h=prod_(out_shape[:-1]))
        elif x.ndim == 3:
            src = metal_kernel.max_3d(x.shape, x.stride, dim=dim if dim != -1 else 2, op=op)
            if dim == 2 or dim == -1:
                pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, op, w=x.shape[-1], h=prod_(x.shape[:-1]))
                threadsPerGrid = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], prod_(x.shape[:-1]), 1)
                threadsPerThreadgroup = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], 1, 1)
            else:
                pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, op, w=out_shape[-1], h=prod_(out_shape[:-1]))
        elif x.ndim == 2:
            src = metal_kernel.max_2d(x.shape, x.stride, dim=dim if dim != -1 else 1, op=op)
            if dim == 1 or dim == -1:
                pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, op, w=x.shape[-1], h=prod_(x.shape[:-1]))
                threadsPerGrid = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], prod_(x.shape[:-1]), 1)
                threadsPerThreadgroup = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], 1, 1)
            else:
                pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, op, w=out_shape[-1], h=prod_(out_shape[:-1]))

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        if dim == -1:
            out_ptr = dispatcher.dispatch(op=UnaryOps.MAX if op == "max" else UnaryOps.SUM, device=Device.CPU, x=Buffer(out_ptr, out_shape, x.device, x.dtype), dim=-1)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.MAX, Device.METAL)
    def max(x: Buffer, dim: int, backward: bool = False, out: Buffer = None):
        return MetalGPU.reduce(x, dim, "max")

    @staticmethod
    @dispatcher.register(UnaryOps.SUM, Device.METAL)
    def sum(x: Buffer, dim: int):
        return MetalGPU.reduce(x, dim, "sum")

    @staticmethod
    @dispatcher.register(UnaryOps.WHERE, Device.METAL)
    def where(x: Buffer, inp: Buffer, other: Buffer):
        out_ptr = CPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        inp_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(inp.ptr, inp.nbytes), inp.nbytes, Metal.MTLResourceStorageModeShared, None)
        other_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(other.ptr, other.nbytes), other.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        src = metal_kernel.where(x.shape, inp=True if inp.ndim == 0 else False, other=True if other.ndim == 0 else False)
        # print(src)
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "where", w=x.shape[-1], h=prod_(x.shape[:-1]))

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(inp_buf, 0, 2)
        computeEncoder.setBuffer_offset_atIndex_(other_buf, 0, 3)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.TRANSPOSE, Device.METAL)
    def transpose(x: Buffer, out_stride: tuple):
        out_ptr = CPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        src = metal_kernel.transpose(x.shape)
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "transpose", w=x.shape[1], h=x.shape[0])

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.NEG, Device.METAL)
    def neg(x: Buffer):
        out_ptr = CPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        src = metal_kernel.utils(x.shape, "neg")
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "neg", w=x.shape[-1], h=prod_(x.shape[:-1]))

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.EXP, Device.METAL)
    def exp(x: Buffer):
        out_ptr = CPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        src = metal_kernel.utils(x.shape, "exp")
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "exp", w=x.shape[-1], h=prod_(x.shape[:-1]))

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.LOG, Device.METAL)
    def log(x: Buffer):
        out_ptr = CPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        src = metal_kernel.utils(x.shape, "log")
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "log", w=x.shape[-1], h=prod_(x.shape[:-1]))

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.POW, Device.METAL)
    def pow(x: Buffer, val: Scalar):
        out_ptr = CPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        src = metal_kernel.utils(x.shape, "pow", val)
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "pow", w=x.shape[-1], h=prod_(x.shape[:-1]))

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.SQRT, Device.METAL)
    def sqrt(x: Buffer):
        out_ptr = CPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        src = metal_kernel.utils(x.shape, "sqrt")
        pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "sqrt", w=x.shape[-1], h=prod_(x.shape[:-1]))

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)

        MetalGPU._run_kernel(commandBuffer, computeEncoder, threadsPerGrid, threadsPerThreadgroup)

        return out_ptr
