# https://dougallj.github.io/applegpu/docs.html

import functools
import math
import struct
import sysconfig

import _allocate  # type: ignore
import Metal
from cffi import FFI

from dlgrad.buffer import Buffer
from dlgrad.device import Device
from dlgrad.dispatch import dispatcher
from dlgrad.dtype import CDataPtr, DType, Scalar
from dlgrad.helpers import BinaryOps, UnaryOps, cal_sum_max_out_shape, prod_


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

sum_metallib_path = f"{sysconfig.get_paths()['purelib']}/dlgrad/src/metal/sum.metallib"
sum_lib = device.newLibraryWithURL_error_(sum_metallib_path, None)[0]

max_metallib_path = f"{sysconfig.get_paths()['purelib']}/dlgrad/src/metal/max.metallib"
max_lib = device.newLibraryWithURL_error_(max_metallib_path, None)[0]

matmul_metallib_path = f"{sysconfig.get_paths()['purelib']}/dlgrad/src/metal/matmul.metallib"
matmul_lib = device.newLibraryWithURL_error_(matmul_metallib_path, None)[0]

transpose_metallib_path = f"{sysconfig.get_paths()['purelib']}/dlgrad/src/metal/transpose.metallib"
transpose_lib = device.newLibraryWithURL_error_(transpose_metallib_path, None)[0]

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

sum2d_func_name = sum_lib.newFunctionWithName_("sum2d")
sum2d_pso = device.newComputePipelineStateWithFunction_error_(sum2d_func_name, None)[0]
sum2d_dim1_func_name = sum_lib.newFunctionWithName_("sum2d_dim1")
sum2d_dim1_pso = device.newComputePipelineStateWithFunction_error_(sum2d_dim1_func_name, None)[0]
sum2d_dim0_func_name = sum_lib.newFunctionWithName_("sum2d_dim0")
sum2d_dim0_pso = device.newComputePipelineStateWithFunction_error_(sum2d_dim0_func_name, None)[0]

max2d_func_name = max_lib.newFunctionWithName_("max2d")
max2d_pso = device.newComputePipelineStateWithFunction_error_(max2d_func_name, None)[0]
max2d_dim1_func_name = max_lib.newFunctionWithName_("max2d_dim1")
max2d_dim1_pso = device.newComputePipelineStateWithFunction_error_(max2d_dim1_func_name, None)[0]

matmul_func_name = matmul_lib.newFunctionWithName_("matmul")
matmul_pso = device.newComputePipelineStateWithFunction_error_(matmul_func_name, None)[0]

transpose_func_name = transpose_lib.newFunctionWithName_("transpose")
transpose_pso = device.newComputePipelineStateWithFunction_error_(transpose_func_name, None)[0]

# TODO OR NOTE: If the tensor dim is less than 32 (warp), getting wrong results for sum, what to do in this case ?
#               Should I move these tensors to cpu or find a fix for this condition ?
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
    def cal_height(width: int, height: int):  # noqa: ANN205
        return min(math.floor(1024 / width), height)

    @staticmethod
    def cal(width: int):  # noqa: ANN205, N805
        match int(math.log10(width)) + 1: # n digits
            case 1:
                return 1, width
            case 2:
                nelements_per_thread = 4
                if width % nelements_per_thread == 0:
                    nthreads_per_threadgroup = width // nelements_per_thread
                else:
                    next_multiple = width + (nelements_per_thread - (width % nelements_per_thread))
                    nthreads_per_threadgroup = next_multiple // nelements_per_thread

                return nelements_per_thread, nthreads_per_threadgroup
            case 3:
                pass
            case 4:
                nelements_per_thread = 4
                if width % nelements_per_thread == 0:
                    nthreads_per_threadgroup = width // nelements_per_thread
                else:
                    next_multiple = width + (nelements_per_thread - (width % nelements_per_thread))
                    nthreads_per_threadgroup = next_multiple // nelements_per_thread

                return nelements_per_thread, nthreads_per_threadgroup
            case 5:
                pass

    @staticmethod
    def _cal_threds_per_threadgroup(pso, xshape: tuple) -> tuple[int]:  # noqa: ANN001
        if len(xshape) == 2:
            w = pso._.threadExecutionWidth     # Warp
            h = pso._.maxTotalThreadsPerThreadgroup / pso._.threadExecutionWidth
            return (w, h, 1)
        elif len(xshape) == 3:
            return (pso._.threadExecutionWidth, 4, (pso._.maxTotalThreadsPerThreadgroup / pso._.threadExecutionWidth) / 4)

    @staticmethod
    def _run(computeEncoder, commandBuffer, threadsPerGrid, threadsPerThreadgroup) -> None:  # noqa: ANN001
        # This function handles non-uniform sizes
        computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadgroup)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

    @staticmethod
    def _binary_op(x: Buffer, y: Buffer | Scalar, pso, xshape, xstride, yshape, ystride) -> CDataPtr:  # noqa: ANN001
        out_ptr = MetalGPU.malloc(num=x.numel)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(y.ptr, y.nbytes), y.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(MetalGPU.ffi.buffer(out_ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        xshape_buf, _ = get_buffer_for_int_array(xshape)
        yshape_buf, _ = get_buffer_for_int_array(yshape)
        xstride_buf, _ = get_buffer_for_int_array(xstride)
        ystride_buf, _ = get_buffer_for_int_array(ystride)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        computeEncoder.setComputePipelineState_(pso)
        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 2)
        computeEncoder.setBuffer_offset_atIndex_(xshape_buf, 0, 3)
        computeEncoder.setBuffer_offset_atIndex_(xstride_buf, 0, 4)
        computeEncoder.setBuffer_offset_atIndex_(yshape_buf, 0, 5)
        computeEncoder.setBuffer_offset_atIndex_(ystride_buf, 0, 6)
        if x.ndim == 2:
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], 1)
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=pso, xshape=x.shape))
        elif x.ndim == 3:
            threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], x.shape[-3])
            threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=pso, xshape=x.shape))

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(BinaryOps.ADD, Device.METAL)
    def add(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        if x.ndim == 3:
            pso = add3d_pso
        elif x.ndim == 2:
            pso = add2d_pso

        return MetalGPU._binary_op(x, y, pso, x.shape, x.stride, y.shape, y.stride)

    @staticmethod
    @dispatcher.register(BinaryOps.SUB, Device.METAL)
    def sub(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        if x.ndim == 3:
            pso = sub3d_pso
        elif x.ndim == 2:
            pso = sub2d_pso

        return MetalGPU._binary_op(x, y, pso, x.shape, x.stride, y.shape, y.stride)

    @staticmethod
    @dispatcher.register(BinaryOps.MUL, Device.METAL)
    def mul(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        if x.ndim == 3:
            pso = mul3d_pso
        elif x.ndim == 2:
            pso = mul2d_pso

        return MetalGPU._binary_op(x, y, pso, x.shape, x.stride, y.shape, y.stride)

    @staticmethod
    @dispatcher.register(BinaryOps.DIV, Device.METAL)
    def div(x: Buffer, y: Buffer | Scalar) -> CDataPtr:
        if x.ndim == 3:
            pso = div3d_pso
        elif x.ndim == 2:
            pso = div2d_pso

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

    @staticmethod
    @dispatcher.register(UnaryOps.SUM, Device.METAL)
    def sum(x: Buffer, dim: int) -> CDataPtr:
        # simd_shuffle_down does not work if warp is not 32
        def next_divisible_by_32(n):  # noqa: ANN001, ANN202
            return n if n % 32 == 0 else n + (32 - n % 32)

        num = prod_(cal_sum_max_out_shape(ndim=x.ndim, dim=dim, inp_shape=x.shape))
        out_ptr = MetalGPU.calloc(num=num)

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()
        if x.ndim == 2:
            if dim == -1:
                nbytes = DType.get_n_bytes(x.dtype)
                out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
                    MetalGPU.ffi.buffer(out_ptr, nbytes), nbytes, Metal.MTLResourceStorageModeShared, None
                )
                computeEncoder.setComputePipelineState_(sum2d_pso)
                threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], 1)
                threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=sum2d_pso, xshape=x.shape))
            elif dim == 1:
                xshape_buf, _ = get_buffer_for_int_array(x.shape)
                nbytes = x.shape[0] * DType.get_n_bytes(x.dtype)
                out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
                    MetalGPU.ffi.buffer(out_ptr, nbytes), nbytes, Metal.MTLResourceStorageModeShared, None
                )
                computeEncoder.setComputePipelineState_(sum2d_dim1_pso)
                computeEncoder.setBuffer_offset_atIndex_(xshape_buf, 0, 2)
                threadsPerGrid = Metal.MTLSizeMake(next_divisible_by_32(x.shape[1]), x.shape[0], 1)
                threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=sum2d_dim1_pso, xshape=x.shape))
            elif dim == 0:
                nbytes = x.shape[1] * DType.get_n_bytes(x.dtype)
                out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
                    MetalGPU.ffi.buffer(out_ptr, nbytes), nbytes, Metal.MTLResourceStorageModeShared, None
                )
                computeEncoder.setComputePipelineState_(sum2d_dim0_pso)
                threadsPerGrid = Metal.MTLSizeMake(x.shape[-1], x.shape[-2], 1)
                threadsPerThreadgroup = Metal.MTLSizeMake(*MetalGPU._cal_threds_per_threadgroup(pso=sum2d_dim0_pso, xshape=x.shape))

        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.MAX, Device.METAL)
    def max(x: Buffer, dim: int) -> CDataPtr:
        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)

        xshape_buf, _ = get_buffer_for_int_array(x.shape)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()
        if x.ndim == 2:
            nelements_per_thread, threadgroup_width = MetalGPU.cal(width=x.shape[-1])

            nelements_per_thread_bytes = struct.pack('i', nelements_per_thread)

            h = MetalGPU.cal_height(width=threadgroup_width, height=x.shape[0])

            threadsPerGrid = Metal.MTLSizeMake(threadgroup_width, x.shape[0], 1) # the total number of threads in the grid
            threadsPerThreadgroup = Metal.MTLSizeMake(threadgroup_width, h, 1)

            xshape_buf, _ = get_buffer_for_int_array(x.shape)

            num_floats = threadgroup_width * h
            threadgroup_memory_bytes = num_floats * 4
            computeEncoder.setThreadgroupMemoryLength_atIndex_(threadgroup_memory_bytes, 0)

            if dim == -1:
                tmp_ptr = MetalGPU.calloc(num=x.shape[0])
                out_tmp_buf = Buffer(data=tmp_ptr, shape=(x.shape[0], 1), dtype=DType.FLOAT32)
                tmp_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
                    MetalGPU.ffi.buffer(out_tmp_buf.ptr, out_tmp_buf.nbytes), out_tmp_buf.nbytes, Metal.MTLResourceStorageModeShared, None
                )
                computeEncoder.setComputePipelineState_(max2d_pso)
            elif dim == 1:
                out_ptr = MetalGPU.calloc(num=x.shape[0])
                tmp_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
                    MetalGPU.ffi.buffer(out_ptr, x.shape[0]*4), x.shape[0]*4, Metal.MTLResourceStorageModeShared, None
                )
                computeEncoder.setComputePipelineState_(max2d_dim1_pso)

        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(tmp_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(xshape_buf, 0, 2)
        computeEncoder.setBytes_length_atIndex_(nelements_per_thread_bytes, len(nelements_per_thread_bytes), 3)

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        if dim == -1:
            out_ptr = dispatcher.dispatch(op=UnaryOps.MAX, device=Device.CPU, x=out_tmp_buf, dim=dim)

        return out_ptr, None

    @staticmethod
    @dispatcher.register(BinaryOps.MATMUL, Device.METAL)
    def matmul(x: Buffer, y: Buffer) -> CDataPtr:
        def next_divisible_by_32(n):  # noqa: ANN001, ANN202
            return n if n % 32 == 0 else n + (32 - n % 32)

        out_ptr = MetalGPU.malloc(num=x.shape[0]*y.shape[1])

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        y_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(y.ptr, y.nbytes), y.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(out_ptr, x.shape[0]*y.shape[1]*4), x.shape[0]*y.shape[1]*4, Metal.MTLResourceStorageModeShared, None)

        xshape_buf, _ = get_buffer_for_int_array(x.shape)
        yshape_buf, _ = get_buffer_for_int_array(y.shape)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        if x.ndim == 2:
            computeEncoder.setComputePipelineState_(matmul_pso)
            output_rows = x.shape[0]
            output_cols = y.shape[1]
            output_rows = next_divisible_by_32(output_rows)
            output_cols = next_divisible_by_32(output_cols)
            threadsPerGrid = Metal.MTLSizeMake(output_cols, output_rows, 1)
            threadsPerThreadgroup = Metal.MTLSizeMake(32, 32, 1)

        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(y_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 2)
        computeEncoder.setBuffer_offset_atIndex_(xshape_buf, 0, 3)
        computeEncoder.setBuffer_offset_atIndex_(yshape_buf, 0, 4)

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        return out_ptr

    @staticmethod
    @dispatcher.register(UnaryOps.TRANSPOSE, Device.METAL)
    def transpose(x: Buffer) -> CDataPtr:
        out_ptr = MetalGPU.malloc(num=prod_(x.shape))

        x_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
           MetalGPU.ffi.buffer(x.ptr, x.nbytes), x.nbytes, Metal.MTLResourceStorageModeShared, None)
        out_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
            MetalGPU.ffi.buffer(out_ptr, prod_(x.shape)*4), prod_(x.shape)*4, Metal.MTLResourceStorageModeShared, None)

        xshape_buf, _ = get_buffer_for_int_array(x.shape)

        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        if x.ndim == 2:
            computeEncoder.setComputePipelineState_(transpose_pso)
            gridWidth = (x.shape[1] + 31) / 32 * 32
            gridHeight = (x.shape[0] + 31) / 32 * 32
            threadsPerGrid = Metal.MTLSizeMake(gridWidth, gridHeight, 1)
            threadsPerThreadgroup = Metal.MTLSizeMake(32, 32, 1)

        computeEncoder.setBuffer_offset_atIndex_(x_buf, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(out_buf, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(xshape_buf, 0, 2)

        MetalGPU._run(computeEncoder=computeEncoder, commandBuffer=commandBuffer, threadsPerGrid=threadsPerGrid, threadsPerThreadgroup=threadsPerThreadgroup)

        return out_ptr
