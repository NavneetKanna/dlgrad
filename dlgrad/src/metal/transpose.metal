#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_compute>
#include <metal_atomic>
#include <metal_math>
using namespace metal;

kernel void transpose(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    device int *xshape [[ buffer(2) ]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tid [[threadgroup_position_in_grid]]
)
{
    int x_row = gid.x;
    int x_col = gid.y;

    // out[col][row] = x[row][col]
    if (x_row < xshape[1] && x_col < xshape[0]) {
        // out[x_row * xshape[0] + x_col] = x[x_col * xshape[1] + x_row];
        out[x_col * xshape[0] + x_row] = x[x_row * xshape[1] + x_col];
    }
}