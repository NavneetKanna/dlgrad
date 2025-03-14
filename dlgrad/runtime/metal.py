# https://dougallj.github.io/applegpu/docs.html

import struct
import sysconfig

import _allocate  # type: ignore
import Metal
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, Scalar
from dlgrad.helpers import BinaryOps

device = Metal.MTLCreateSystemDefaultDevice()
commandQueue = device.newCommandQueue()  # noqa: N816

arithmetic_metallib_path = f"{sysconfig.get_paths()['purelib']}/dlgrad/src/metal/arithmetic.metallib"
arithmetic_lib = device.newLibraryWithURL_error_(arithmetic_metallib_path, None)[0]
add_func_name = arithmetic_lib.newFunctionWithName_("add_arrays")
add_pso = device.newComputePipelineStateWithFunction_error_(add_func_name, None)[0]

sub_func_name = arithmetic_lib.newFunctionWithName_("sub_arrays")
sub_pso = device.newComputePipelineStateWithFunction_error_(add_func_name, None)[0]

mul_func_name = arithmetic_lib.newFunctionWithName_("mul_arrays")
mul_pso = device.newComputePipelineStateWithFunction_error_(add_func_name, None)[0]

div_func_name = arithmetic_lib.newFunctionWithName_("div_arrays")
div_pso = device.newComputePipelineStateWithFunction_error_(add_func_name, None)[0]

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
    def _run(pso, x_buf, y_buf, out_buf) -> None:  # noqa: ANN001
        commandBuffer = commandQueue.commandBuffer()  # noqa: N806
        computeEncoder = commandBuffer.computeCommandEncoder()  # noqa: N806

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 2)

        threadsPerThreadgroup = Metal.MTLSizeMake(1024, 1, 1)  # noqa: N806
        threadgroupSize = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)  # noqa: N806

        computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerThreadgroup, threadgroupSize)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return

    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.METAL)
    def add(x: Buffer, y: Buffer | Scalar):  # noqa: ANN205
        out_ptr = MetalGPU.malloc(num=x.numel)
        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(x.ptr, x.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(y.ptr, y.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(out_ptr, x.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501

        MetalGPU._run(pso=add_pso, x_buf=x_buf, y_buf=y_buf, out_buf=out_buf)

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.METAL)
    def sub(x: Buffer, y: Buffer | Scalar):  # noqa: ANN205
        out_ptr = MetalGPU.malloc(num=x.numel)
        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(x.ptr, x.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(y.ptr, y.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(out_ptr, x.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501

        MetalGPU._run(pso=sub_pso, x_buf=x_buf, y_buf=y_buf, out_buf=out_buf)

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.METAL)
    def mul(x: Buffer, y: Buffer | Scalar):  # noqa: ANN205
        out_ptr = MetalGPU.malloc(num=x.numel)
        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(x.ptr, x.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(y.ptr, y.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(out_ptr, x.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501

        MetalGPU._run(pso=mul_pso, x_buf=x_buf, y_buf=y_buf, out_buf=out_buf)

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.METAL)
    def div(x: Buffer, y: Buffer | Scalar):  # noqa: ANN205
        out_ptr = MetalGPU.malloc(num=x.numel)
        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(x.ptr, x.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(y.ptr, y.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(out_ptr, x.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501

        MetalGPU._run(pso=div_pso, x_buf=x_buf, y_buf=y_buf, out_buf=out_buf)
