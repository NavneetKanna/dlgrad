#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_compute>
#include <metal_atomic>
#include <metal_math>
using namespace metal;

// Tiled transpose
kernel void transpose(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    device int *xshape [[ buffer(2) ]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tid [[threadgroup_position_in_grid]]
)
{
    threadgroup float x_shared[32][32];

    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;

    // Load a block to the shared memory
    for (int i=0; i<xshape[1]; i+=32) {
        if (row < xshape[0] && i + lid.x < xshape[1]) {
            x_shared[lid.y][lid.x] = x[row*xshape[1] + i + lid.x];
        } else {
            x_shared[lid.y][lid.x] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read a col from the shared memory and write it to a row in the output matrix
    if (row < xshape[0] && col < xshape[1]) {
        out[row * xshape[0] + col] = x_shared[col][row];
    }
}