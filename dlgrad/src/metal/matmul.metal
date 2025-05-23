#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_compute>
#include <metal_atomic>
#include <metal_math>
using namespace metal;


/*
// Tiled matmul
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
        if (row < xshape[0] && i + lid.x < xshape[1]) {
            x_shared[lid.y][lid.x] = x[row*xshape[1] + i + lid.x];
        } else {
            x_shared[lid.y][lid.x] = 0.0f;
        }
        if (col < yshape[1] && i + lid.y < yshape[0]) {
            y_shared[lid.y][lid.x] = y[(i + lid.y) * yshape[1] + col];
        } else {
            y_shared[lid.y][lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < 32; j++) {
            sum += x_shared[lid.y][j] * y_shared[j][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < xshape[0] && col < yshape[1]) {
        out[row * yshape[1] + col] = sum;
    }
}
*/

kernel void matmul(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* xshape [[buffer(3)]],
    device const int* yshape [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tid [[threadgroup_position_in_grid]]
) {
    threadgroup float x_shared[32][32];
    threadgroup float y_shared[32][32];

    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;

    const uint K = xshape[1];
    for (uint i = 0; i < K; i += 32) {
        // Load tile from x into x_shared
        uint x_row = row;
        uint x_col = i + lid.x;
        if (x_row < xshape[0] && x_col < K) {
            x_shared[lid.y][lid.x] = x[x_row * K + x_col];
        } else {
            x_shared[lid.y][lid.x] = 0.0f;
        }

        // Load tile from y into y_shared
        uint y_row = i + lid.y;
        uint y_col = col;
        if (y_row < yshape[0] && y_col < yshape[1]) {
            y_shared[lid.y][lid.x] = y[y_row * yshape[1] + y_col];
        } else {
            y_shared[lid.y][lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial sum
        for (uint j = 0; j < 32; ++j) {
            sum += x_shared[lid.y][j] * y_shared[j][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result to output
    if (row < xshape[0] && col < yshape[1]) {
        out[row * yshape[1] + col] = sum;
    }
}