# https://dougallj.github.io/applegpu/docs.html

import functools
import struct
import sysconfig

import _allocate  # type: ignore
import Metal
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, Scalar
from dlgrad.helpers import BinaryOps, UnaryOps, prod_


# TODO: Maybe create buffers during creation time ?
@functools.cache
def get_buffer_for_int_array(arr: tuple) -> any:
    ffi_arr = MetalGPU.ffi.new("int[]", list(arr))
    size = len(arr) * MetalGPU.ffi.sizeof("int")
    buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
        MetalGPU.ffi.buffer(ffi_arr), size, Metal.MTLResourceStorageModeShared, None)
    return buf, ffi_arr


device = Metal.MTLCreateSystemDefaultDevice()
commandQueue = device.newCommandQueue()

# threadExecutionWidth - warp in cuda
# make the number of threads in the threadgroup a multiple of threadExecutionWidth.
# maxTotalThreadsPerThreadgroup - 1024 threadExecutionWidth - 32

# TODO: Can i create only 1 pso ?
arithmetic_metallib_path = f"{sysconfig.get_paths()['purelib']}/dlgrad/src/metal/arithmetic.metallib"
arithmetic_lib = device.newLibraryWithURL_error_(arithmetic_metallib_path, None)[0]

utils_metallib_path = f"{sysconfig.get_paths()['purelib']}/dlgrad/src/metal/utils.metallib"
utils_lib = device.newLibraryWithURL_error_(utils_metallib_path, None)[0]

add2d_func_name = arithmetic_lib.newFunctionWithName_("add2d")
add2d_pso = device.newComputePipelineStateWithFunction_error_(add2d_func_name, None)[0]
add3d_func_name = arithmetic_lib.newFunctionWithName_("add3d")
add3d_pso = device.newComputePipelineStateWithFunction_error_(add3d_func_name, None)[0]

sub2d_func_name = arithmetic_lib.newFunctionWithName_("sub2d")
sub2d_pso = device.newComputePipelineStateWithFunction_error_(sub2d_func_name, None)[0]
sub3d_func_name = arithmetic_lib.newFunctionWithName_("sub3d")
sub3d_pso = device.newComputePipelineStateWithFunction_error_(sub3d_func_name, None)[0]

mul2d_func_name = arithmetic_lib.newFunctionWithName_("mul2d")
mul2d_pso = device.newComputePipelineStateWithFunction_error_(mul2d_func_name, None)[0]
mul3d_func_name = arithmetic_lib.newFunctionWithName_("mul3d")
mul3d_pso = device.newComputePipelineStateWithFunction_error_(mul3d_func_name, None)[0]

div2d_func_name = arithmetic_lib.newFunctionWithName_("div2d")
div2d_pso = device.newComputePipelineStateWithFunction_error_(div2d_func_name, None)[0]
div3d_func_name = arithmetic_lib.newFunctionWithName_("div3d")
div3d_pso = device.newComputePipelineStateWithFunction_error_(div3d_func_name, None)[0]

add_func_name = arithmetic_lib.newFunctionWithName_("add")
add_pso = device.newComputePipelineStateWithFunction_error_(add_func_name, None)[0]

sub_func_name = arithmetic_lib.newFunctionWithName_("sub")
sub_pso = device.newComputePipelineStateWithFunction_error_(sub_func_name, None)[0]

mul_func_name = arithmetic_lib.newFunctionWithName_("mul")
mul_pso = device.newComputePipelineStateWithFunction_error_(mul_func_name, None)[0]

div_func_name = arithmetic_lib.newFunctionWithName_("div")
div_pso = device.newComputePipelineStateWithFunction_error_(div_func_name, None)[0]

neg2d_func_name = utils_lib.newFunctionWithName_("mneg2d")
neg2d_pso = device.newComputePipelineStateWithFunction_error_(neg2d_func_name, None)[0]
neg3d_func_name = utils_lib.newFunctionWithName_("mneg3d")
neg3d_pso = device.newComputePipelineStateWithFunction_error_(neg3d_func_name, None)[0]

log2d_func_name = utils_lib.newFunctionWithName_("mlog2d")
log2d_pso = device.newComputePipelineStateWithFunction_error_(log2d_func_name, None)[0]
log3d_func_name = utils_lib.newFunctionWithName_("mlog3d")
log3d_pso = device.newComputePipelineStateWithFunction_error_(log3d_func_name, None)[0]

exp2d_func_name = utils_lib.newFunctionWithName_("mexp2d")
exp2d_pso = device.newComputePipelineStateWithFunction_error_(exp2d_func_name, None)[0]
exp3d_func_name = utils_lib.newFunctionWithName_("mexp3d")
exp3d_pso = device.newComputePipelineStateWithFunction_error_(exp3d_func_name, None)[0]

sqrt2d_func_name = utils_lib.newFunctionWithName_("msqrt2d")
sqrt2d_pso = device.newComputePipelineStateWithFunction_error_(sqrt2d_func_name, None)[0]
sqrt3d_func_name = utils_lib.newFunctionWithName_("msqrt3d")
sqrt3d_pso = device.newComputePipelineStateWithFunction_error_(sqrt3d_func_name, None)[0]

pow2d_func_name = utils_lib.newFunctionWithName_("mpow2d")
pow2d_pso = device.newComputePipelineStateWithFunction_error_(pow2d_func_name, None)[0]
pow3d_func_name = utils_lib.newFunctionWithName_("mpow3d")
pow3d_pso = device.newComputePipelineStateWithFunction_error_(pow3d_func_name, None)[0]

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
    def _cal_threds_per_threadgroup(pso, xshape: tuple) -> tuple[int]:  # noqa: ANN001
        if len(xshape) == 2:
            return (pso._.threadExecutionWidth, pso._.maxTotalThreadsPerThreadgroup / pso._.threadExecutionWidth, 1)
        elif len(xshape) == 3:
            return (pso._.threadExecutionWidth, 4, (pso._.maxTotalThreadsPerThreadgroup / pso._.threadExecutionWidth) / 4)

    @staticmethod
    def _run(computeEncoder, commandBuffer, threadsPerGrid, threadsPerThreadgroup) -> None:  # noqa: ANN001
        # Therefore the num of threadgroups = threadsPerGrid / threadsPerThreadgroup
        # This function handles non-uniform sizes
        computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadgroup)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

    @staticmethod
    def _binary_op(x: Buffer, y: Buffer | Scalar, pso, xshape, xstride, yshape, ystride) -> CDataPtr:  # noqa: ANN001
        out_ptr = MetalGPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(y.ptr, y.nbytes), y.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        # TODO: Is this right for same shape ?
        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 2)
        if xshape != yshape:
            xshape_buf = get_buffer_for_int_array(xshape)
            yshape_buf = get_buffer_for_int_array(yshape)
            xstride_buf = get_buffer_for_int_array(xstride)
            ystride_buf = get_buffer_for_int_array(ystride)

            computeEncoder.setBuffer_offset_atIndex_(xshape_buf, 0, 3)
            computeEncoder.setBuffer_offset_atIndex_(xstride_buf, 0, 4)
            computeEncoder.setBuffer_offset_atIndex_(yshape_buf, 0, 5)
            computeEncoder.setBuffer_offset_atIndex_(ystride_buf, 0, 6)

            if len(xshape) == 3:    # 3D
                # The total num of threads (w, h, d)
                threadsPerGrid = Metal.MTLSizeMake(xshape[-1], xshape[-2], xshape[-3])
            elif len(xshape) == 2:  # 2D
                # The total num of threads (w, h)
                threadsPerGrid = Metal.MTLSizeMake(xshape[-1], xshape[-2], 1)
        else:
            # The total num of threads (tensor numel, 1, 1)
            threadsPerGrid = Metal.MTLSizeMake(prod_(xshape), 1, 1)

        # Num of threads per threadgroup
        threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=pso, xshape=xshape, yshape=yshape))

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.METAL)
    def add(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        if x.ndim == 3:
            pso = add3d_pso
        elif x.ndim == 2:
            pso = add2d_pso

        if x.shape == y.shape:
            pso = add_pso
        return MetalGPU._binary_op(x, y, pso, x.shape, x.stride, y.shape, y.stride)

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.METAL)
    def sub(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        if x.ndim == 3:
            pso = sub3d_pso
        elif x.ndim == 2:
            pso = sub2d_pso

        if x.shape == y.shape:
            pso = sub_pso
        return MetalGPU._binary_op(x, y, pso, x.shape, x.stride, y.shape, y.stride)

    @staticmethod
    @dispatcher.register(BinaryOps.MUL, Device.METAL)
    def mul(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        if x.ndim == 3:
            pso = mul3d_pso
        elif x.ndim == 2:
            pso = mul2d_pso

        if x.shape == y.shape:
            pso = mul_pso
        return MetalGPU._binary_op(x, y, pso, x.shape, x.stride, y.shape, y.stride)

    @staticmethod
    @dispatcher.register(BinaryOps.DIV, Device.METAL)
    def div(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        if x.ndim == 3:
            pso = div3d_pso
        elif x.ndim == 2:
            pso = div2d_pso

        if x.shape == y.shape:
            pso = div_pso
        return MetalGPU._binary_op(x, y, pso, x.shape, x.stride, y.shape, y.stride)

    @staticmethod
    @dispatcher.register(UnaryOps.NEG, Device.METAL)
    def neg(x: Buffer) -> CDataPtr:
        out_ptr = MetalGPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        xstride_buf, _ = get_buffer_for_int_array(x.stride)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        if x.ndim == 2:
            computeEncoder.setComputePipelineState_(neg2d_pso)
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], 1)
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=neg2d_pso, xshape=x.shape))
        elif x.ndim == 3:
            computeEncoder.setComputePipelineState_(neg3d_pso)
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], x.shape[-3])
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=neg3d_pso, xshape=x.shape))

        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(xstride_buf, 0, 2)

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.LOG, Device.METAL)
    def log(x: Buffer) -> CDataPtr:
        out_ptr = MetalGPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        xstride_buf, _ = get_buffer_for_int_array(x.stride)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        if x.ndim == 2:
            computeEncoder.setComputePipelineState_(log2d_pso)
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], 1)
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=log2d_pso, xshape=x.shape))
        elif x.ndim == 3:
            computeEncoder.setComputePipelineState_(log3d_pso)
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], x.shape[-3])
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=log3d_pso, xshape=x.shape))

        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(xstride_buf, 0, 2)

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.EXP, Device.METAL)
    def exp(x: Buffer) -> CDataPtr:
        out_ptr = MetalGPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        xstride_buf, _ = get_buffer_for_int_array(x.stride)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        if x.ndim == 2:
            computeEncoder.setComputePipelineState_(exp2d_pso)
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], 1)
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=exp2d_pso, xshape=x.shape))
        elif x.ndim == 3:
            computeEncoder.setComputePipelineState_(exp3d_pso)
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], x.shape[-3])
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=exp3d_pso, xshape=x.shape))

        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(xstride_buf, 0, 2)

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.SQRT, Device.METAL)
    def sqrt(x: Buffer) -> CDataPtr:
        out_ptr = MetalGPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        xstride_buf, _ = get_buffer_for_int_array(x.stride)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        if x.ndim == 2:
            computeEncoder.setComputePipelineState_(sqrt2d_pso)
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], 1)
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=sqrt2d_pso, xshape=x.shape))
        elif x.ndim == 3:
            computeEncoder.setComputePipelineState_(sqrt3d_pso)
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], x.shape[-3])
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=sqrt3d_pso, xshape=x.shape))

        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(xstride_buf, 0, 2)

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.POW, Device.METAL)
    def pow(x: Buffer, val: Scalar) -> CDataPtr:
        out_ptr = MetalGPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        xstride_buf, _ = get_buffer_for_int_array(x.stride)
        ffi_arr = MetalGPU.ffi.new("float[]", [val])
        size = MetalGPU.ffi.sizeof("float")
        val_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(ffi_arr), size, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        if x.ndim == 2:
            computeEncoder.setComputePipelineState_(pow2d_pso)
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], 1)
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=pow2d_pso, xshape=x.shape))
        elif x.ndim == 3:
            computeEncoder.setComputePipelineState_(pow3d_pso)
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], x.shape[-3])
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=pow3d_pso, xshape=x.shape))

        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(val_buf, 0, 2)
        computeEncoder.setBuffer_offset_atIndex_(xstride_buf, 0, 3)

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        return out_ptr
