#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_compute>
#include <metal_atomic>
using namespace metal;

kernel void sum2d_dim1(device const float *x, device atomic_float *out,
                      uint2 tid [[thread_position_in_grid]], uint2 grid_size [[threads_per_grid]], 
                      uint simd_size [[threads_per_simdgroup]], uint simd_lane_id [[thread_index_in_simdgroup]])
{
    uint index = tid.y * grid_size.x + tid.x;
    float val = x[index];

    // Perform reduction for each warp
    for (uint offset=simd_size/2; offset>0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }

    // If it is the first thread in the simd/warp group,
    // then update the output tensor with val by addition
    // For example, if the width of the tensor is 64 and 
    // thread group width is 32, then for the first row
    // there will be 2 val's that needs to be added, one,
    // from the first warp and two, from the second
    // warp. So after say the first warp completes, fetch
    // the value of output at index grid_size.x, add val to it and
    // write back, now after the second warp completes, 
    // fetch the value of output at the same index ..., which
    // now contains the val from the other warp, add to it and 
    // write back. So therefore, output[grid_size.x] will have the sum of
    // the first row of the input
    if (simd_lane_id == 0) {
        atomic_fetch_add_explicit(&out[tid.y], val, memory_order_relaxed);
    }
}






/*

array = 64x64
warp = 32
thread block = 32x32

offset = 16

[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ]            [ 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ]
  |                                                                    |
  |--------------------------------------------------------------------
  |
[ 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46 ]        -> remember that these are all val variables in the individual threads


offset = 8

[ 16, 18, 20, 22, 24, 26, 28, 30 ]                 [ 32, 34, 36, 38, 40, 42, 44, 46 ]
  |                                                   |
  |---------------------------------------------------
  |  
[ 48, 52, 56, 60, 64, 68, 72, 76 ]


offset = 4

[48, 52, 56, 60 ]         [ 64, 68, 72, 76 ]
  |                         |
  |-------------------------
  |
[ 112, 120, 128, 136 ]


offset = 2

[ 112, 120 ]        [ 128, 136 ]
  |                    |
  |--------------------
  |
[ 240, 256 ]


offset = 1

[ 240 ]    [ 256 ]
  |           |
  |-----------
  |
[ 496 ]               -> The val variable of thread index 0 will hold the final sum


*/


