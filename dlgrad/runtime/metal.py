# ruff: noqa

import struct
from functools import cache
import os
import math
import Metal
from cffi import FFI
import pathlib
from dlgrad.buffer import Buffer
from dlgrad.codegen import metal_kernel, cpu_kernel
from dlgrad.runtime.cpu import CPU
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, DType
from dlgrad.helpers import CACHE_DIR, BinaryOps, UnaryOps, cal_sum_max_out_shape, prod_
from dlgrad.buffer import Buffer
from dlgrad.dtype import Scalar

device = Metal.MTLCreateSystemDefaultDevice()
commandQueue = device.newCommandQueue()

# NOTE: Metal only supports certain ops on certain dimensions, this is because coming up with an algorithm
# for cases like reduction along dim 1 of a 4d tensor is quite hard and not really used in real world training.
# Hence, I have decided to opt such cases out and only support them on CPU. Metal supports matmul, transpose, reduction
# of full tensor or along rows, etc, where it makes sense to use a GPU.
class MetalGPU:
    """
    Main GPU runtime class for apple silicon gpus which handles the logic of using metal.

    This class uses PyObjC to interact with metal code.
    """
    ffi = FFI()

    @staticmethod
    def malloc(num: int, size: int = struct.calcsize('f')) -> CDataPtr:
        c_code, cdef = cpu_kernel.uninitialized_memory()
        c_code2, cdef2 = cpu_kernel.free_ptr()

        key = CPU._hash_code(c_code)
        key2 = CPU._hash_code(c_code2)
        so_fp = pathlib.Path(CACHE_DIR) / f"unintialized_memory_{key}.so"
        so_fp2 = pathlib.Path(CACHE_DIR) / f"free_{key2}.so"

        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)
        if not os.path.exists(so_fp2):
            CPU._build_shared_object(c_code2, so_fp2)

        lib = CPU._get_handle(str(so_fp))
        lib2 = CPU._get_handle(str(so_fp2))

        CPU._ensure_sig(cdef)
        CPU._ensure_sig(cdef2)

        ptr = CPU.ffi.gc(lib.uninitialized_memory(num*size), lib2.free_ptr)

        if ptr == CPU.ffi.NULL:
            raise MemoryError(f"Failed to allocate {num * size} bytes of memory")

        return ptr

    @staticmethod
    def calloc(num: int, size: int = struct.calcsize('f')) -> CDataPtr:
        c_code, cdef = cpu_kernel.initialized_memory()
        c_code2, cdef2 = cpu_kernel.free_ptr()

        key = CPU._hash_code(c_code)
        key2 = CPU._hash_code(c_code2)
        so_fp = pathlib.Path(CACHE_DIR) / f"initialized_memory_{key}.so"
        so_fp2 = pathlib.Path(CACHE_DIR) / f"free_{key2}.so"

        if not os.path.exists(so_fp):
            CPU._build_shared_object(c_code, so_fp)
        if not os.path.exists(so_fp2):
            CPU._build_shared_object(c_code2, so_fp2)

        lib = CPU._get_handle(str(so_fp))
        lib2 = CPU._get_handle(str(so_fp2))

        CPU._ensure_sig(cdef)
        CPU._ensure_sig(cdef2)

        ptr = CPU.ffi.gc(lib.initialized_memory(num, size), lib2.free_ptr)

        if ptr == CPU.ffi.NULL:
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
        if x.ndim == 3:
            if x.shape[0] != 1:
                out_numel = x.shape[0]*x.shape[1]*y.shape[2]
            else:
                out_numel = y.shape[0]*x.shape[1]*y.shape[2]

        if x.ndim == 3:
            out_ptr = CPU.init_with_scalar(num=out_numel, scalar=0)
        else:
            out_ptr = CPU.init_with_scalar(num=x.shape[0]*y.shape[1], scalar=0)

        if x.ndim == 3:
            x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
            y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(y.ptr, y.nbytes), y.nbytes, Metal.MTLResourceStorageModeShared, None)
            out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, out_numel*4), out_numel*4, Metal.MTLResourceStorageModeShared, None)
        else:
            x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
            y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(y.ptr, y.nbytes), y.nbytes, Metal.MTLResourceStorageModeShared, None)
            out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.shape[0]*y.shape[1]*4), x.shape[0]*y.shape[1]*4, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        if x.ndim == 3:
            broadcast_x, broadcast_y = False, False
            if x.shape[0] == 1:
                broadcast_x = True
            if y.shape[0] == 1:
                broadcast_y = True
            src = metal_kernel.matmul_3d(x.shape[0] if x.shape[0] != 1 else y.shape[0], x.shape[1], x.shape[2], y.shape[2], broadcast_x, broadcast_y)
            # TODO: Add a new function to return only pso and another to choose from threadgroup or thread dispatch
            pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "matmul", w=32 * (y.shape[1]/8), h=32*(x.shape[0]/8))
            threadgroupsPerGrid = Metal.MTLSizeMake(math.ceil(y.shape[2]/32), math.ceil(x.shape[1]/32), x.shape[0] if x.shape[0] != 1 else y.shape[0])
            threadsPerThreadgroup = Metal.MTLSizeMake(32, 32, 1)
        else:
            # src = metal_kernel.matmul_fast(x.shape, y.shape)
            # # TODO: Add a new function to return only pso and another to choose from threadgroup or thread dispatch
            # pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "matmul", w=32 * (y.shape[1]/8), h=32*(x.shape[0]/8))
            # threadgroupsPerGrid = Metal.MTLSizeMake(math.ceil(y.shape[1]/8), math.ceil(x.shape[0]/8), 1)
            # threadsPerThreadgroup = Metal.MTLSizeMake(32, 1, 1)

            src = metal_kernel.matmul(x.shape, y.shape)
            # TODO: Add a new function to return only pso and another to choose from threadgroup or thread dispatch
            pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "matmul", w=32 * (y.shape[1]/8), h=32*(x.shape[0]/8))
            threadgroupsPerGrid = Metal.MTLSizeMake(math.ceil(y.shape[1]/32), math.ceil(x.shape[0]/32), 1)
            threadsPerThreadgroup = Metal.MTLSizeMake(32, 32, 1)

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 2)

        computeEncoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroupsPerGrid, threadsPerThreadgroup)
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

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
            src = metal_kernel.reduce_along_rows(x.shape, op=op)
            if dim == 3 or dim == -1:
                pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, op, w=x.shape[-1], h=prod_(x.shape[:-1]))
                threadsPerGrid = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], prod_(x.shape[:-1]), 1)
                threadsPerThreadgroup = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], 1, 1)
        elif x.ndim == 3:
            src = metal_kernel.reduce_along_rows(x.shape, op=op)
            if dim == 2 or dim == -1:
                pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, op, w=x.shape[-1], h=prod_(x.shape[:-1]))
                threadsPerGrid = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], prod_(x.shape[:-1]), 1)
                threadsPerThreadgroup = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], 1, 1)
        elif x.ndim == 2:
            src = metal_kernel.reduce_along_rows(x.shape, op=op)
            if dim == 1 or dim == -1:
                pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, op, w=x.shape[-1], h=prod_(x.shape[:-1]))
                threadsPerGrid = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], prod_(x.shape[:-1]), 1)
                threadsPerThreadgroup = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup() if x.shape[-1] > pso.maxTotalThreadsPerThreadgroup() else x.shape[-1], 1, 1)

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
    def transpose(x: Buffer, out_stride: tuple, dim0: int, dim1: int, out_shape: tuple):
        out_ptr = CPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        if x.ndim == 4 and ((dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1)):
            src = metal_kernel.transpose_4d_12(x.shape, x.stride, out_stride)
            print(src)
            print(x.shape, out_stride, x.stride)
            pso, _, _ = MetalGPU.build_2d_pipeline(src, "transpose", w=1, h=1)
            # Grid: x=W, y=H, z=N*C
            w_threads = math.ceil(x.shape[3] / 32)
            threadgroupsPerGrid = Metal.MTLSizeMake(w_threads, x.shape[2], x.shape[0] * x.shape[1])
            threadsPerThreadgroup = Metal.MTLSizeMake(32, 1, 1)
            print(threadgroupsPerGrid, threadsPerThreadgroup)
            computeEncoder.setComputePipelineState_(pso)
            computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
            computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
            computeEncoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroupsPerGrid, threadsPerThreadgroup)
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return out_ptr
        if x.ndim == 3 and ((dim0 == 0 and dim1 == 1) or (dim0 == 1 and dim1 == 0)):
            src = metal_kernel.transpose_3d_01(x.shape, x.stride)
            pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "transpose", w=x.shape[2], h=x.shape[0]*x.shape[1])
        elif x.ndim == 3 and ((dim0 == 1 and dim1 == 2) or (dim0 == 2 and dim1 == 1)):
            src = metal_kernel.transpose_3d_12(x.shape)
            pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "transpose", w=x.shape[2], h=x.shape[0]*x.shape[1])
            threadgroupsPerGrid = Metal.MTLSizeMake(math.ceil(x.shape[2]/32), math.ceil(x.shape[1]/32), x.shape[0])
            threadsPerThreadgroup = Metal.MTLSizeMake(32, 32, 1)
            computeEncoder.setComputePipelineState_(pso)
            computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
            computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
            computeEncoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroupsPerGrid, threadsPerThreadgroup)
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return out_ptr
        else:
            src = metal_kernel.transpose_2d(x.shape)
            pso, threadsPerGrid, threadsPerThreadgroup = MetalGPU.build_2d_pipeline(src, "transpose", w=x.shape[1], h=x.shape[0])
            threadgroupsPerGrid = Metal.MTLSizeMake(math.ceil(x.shape[1]/32), math.ceil(x.shape[0]/32), 1)
            threadsPerThreadgroup = Metal.MTLSizeMake(32, 32, 1)
            computeEncoder.setComputePipelineState_(pso)
            computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
            computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
            computeEncoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroupsPerGrid, threadsPerThreadgroup)
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return out_ptr

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
