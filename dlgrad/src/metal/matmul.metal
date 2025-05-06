#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_compute>
#include <metal_atomic>
#include <metal_math>
using namespace metal;


kernel void matmul(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* out [[buffer(2)]],
    device int *xshape [[ buffer(3) ]],
    device int *yshape [[ buffer(4) ]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tid [[threadgroup_position_in_grid]]
)
{
    threadgroup float x_shared[32][32];
    threadgroup float y_shared[32][32];

    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;

    for (int i=0; i<xshape[1]; i+=32) {
        x_shared[lid.y][lid.x] = x[row*xshape[1] + i + lid.x];
        y_shared[lid.y][lid.x] = y[(i + lid.y) * yshape[1] + col];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < 32; j++) {
            sum += x_shared[lid.y][j] * y_shared[j][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[row * yshape[1] + col] = sum;
}