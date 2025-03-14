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

add_metallib_path = f"{sysconfig.get_paths()['purelib']}/dlgrad/src/metal/add.metallib"
add_lib = device.newLibraryWithURL_error_(add_metallib_path, None)[0]
add_func_name = add_lib.newFunctionWithName_("add_arrays")
add_pso = device.newComputePipelineStateWithFunction_error_(add_func_name, None)[0]


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
    @dispatcher.register(BinaryOps.ADD, Device.METAL)
    def add(x: Buffer, y: Buffer | Scalar):  # noqa: ANN205
        out_ptr = MetalGPU.malloc(num=x.numel)
        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(x.ptr, x.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(y.ptr, y.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(out_ptr, x.nbytes, Metal.MTLResourceStorageModeShared, None)  # noqa: E501

        commandBuffer = commandQueue.commandBuffer()  # noqa: N806
        computeEncoder = commandBuffer.computeCommandEncoder()  # noqa: N806

        computeEncoder.setComputePipelineState_(add_pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 2)

        threadsPerThreadgroup = Metal.MTLSizeMake(1024, 1, 1)  # noqa: N806
        threadgroupSize = Metal.MTLSizeMake(add_pso.maxTotalThreadsPerThreadgroup(), 1, 1)  # noqa: N806


        computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerThreadgroup, threadgroupSize)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return out_ptr
