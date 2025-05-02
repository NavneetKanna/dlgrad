#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_compute>
#include <metal_atomic>
#include <metal_math>
using namespace metal;
// TODO: Atomic function is too slow



kernel void max2d(
    device const float *x [[ buffer(0) ]],
    device float *out_tmp [[ buffer(1) ]],
    device int *xshape [[ buffer(2) ]],
    constant int& nelements_per_thread_buf [[ buffer(3) ]],
    uint2 threadgroup_size [[ threads_per_threadgroup ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 grid_size [[threads_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    threadgroup float* tmp [[ threadgroup(0) ]]
)
{
    int nelements_per_thread = nelements_per_thread_buf;

    uint index = tid.y * grid_size.x + tid.x;
    int start_idx = index * nelements_per_thread;

    float acc = ((float)(-INFINITY));
    for (int i=0; i<nelements_per_thread; i++) {
        int real_idx = start_idx + i;
        if (real_idx < xshape[1]*xshape[0]) {
            float val = x[real_idx];
            acc = max(val, acc);
        }
    }
    tmp[index] = acc;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.x == 0) {
        float acc = ((float)(-INFINITY));
        uint start = tid.y * threadgroup_size.x;
        uint end = (start + threadgroup_size.x);
        
        for (uint i=start; i<end; i++) {
            float val = tmp[i];
            acc = max(val, acc);
        }

        out_tmp[tid.y] = acc;
   }
}

kernel void max2d_dim1(
    device const float *x [[ buffer(0) ]],
    device float *out_tmp [[ buffer(1) ]],
    device int *xshape [[ buffer(2) ]],
    constant int& nelements_per_thread_buf [[ buffer(3) ]],
    uint2 threadgroup_size [[ threads_per_threadgroup ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 grid_size [[threads_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    threadgroup float* tmp [[ threadgroup(0) ]]
)
{
    int nelements_per_thread = nelements_per_thread_buf;

    uint index = tid.y * grid_size.x + tid.x;
    int start_idx = index * nelements_per_thread;

    float acc = ((float)(-INFINITY));
    for (int i=0; i<nelements_per_thread; i++) {
        int real_idx = start_idx + i;
        if (real_idx < xshape[1]*xshape[0]) {
            float val = x[real_idx];
            acc = max(val, acc);
        }
    }
    tmp[index] = acc;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.x == 0) {
        float acc = ((float)(-INFINITY));
        uint start = tid.y * threadgroup_size.x;
        uint end = (start + threadgroup_size.x);
        
        for (uint i=start; i<end; i++) {
            float val = tmp[i];
            acc = max(val, acc);
        }

        out_tmp[tid.y] = acc;
   }
}
