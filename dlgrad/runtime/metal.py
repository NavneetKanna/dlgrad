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
commandQueue = device.newCommandQueue()

# threadExecutionWidth - warp in cuda
# more no of threadgroups, more time it takes to run
# make the number of threads in the threadgroup a multiple of threadExecutionWidth.
# maxTotalThreadsPerThreadgroup - 1024 threadExecutionWidth - 32

# TODO: Can i create only 1 pso ?
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
        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 2)

        # TODO: If array size is < maxTotalThreadsPerThreadgroup, then set threadsPerThreadgroup = array size
        # TODO: 3D ?
        # TODO: Broadcasting
        # The total num of threads
        threadsPerGrid = Metal.MTLSizeMake(500, 500, 1)
        # Num of threads per threadgroup
        threadsPerThreadgroup = Metal.MTLSizeMake(pso._.maxTotalThreadsPerThreadgroup, pso._.maxTotalThreadsPerThreadgroup, 1)  # noqa: E501
        computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadgroup)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

    @staticmethod
    def _binary_op(x: Buffer, y: Buffer | Scalar, pso) -> CDataPtr:  # noqa: ANN001
        out_ptr = MetalGPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(y.ptr, y.nbytes), y.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        MetalGPU._run(pso=pso, x_buf=x_buf, y_buf=y_buf, out_buf=out_buf)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.METAL)
    def add(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        return MetalGPU._binary_op(x, y, add_pso)

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.METAL)
    def sub(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        return MetalGPU._binary_op(x, y, sub_pso)

    @staticmethod
    @dispatcher.register(BinaryOps.MUL, Device.METAL)
    def mul(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        return MetalGPU._binary_op(x, y, mul_pso)

    @staticmethod
    @dispatcher.register(BinaryOps.DIV, Device.METAL)
    def div(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        return MetalGPU._binary_op(x, y, div_pso)
