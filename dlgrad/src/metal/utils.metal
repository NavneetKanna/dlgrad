#include <metal_math>
using namespace metal;

kernel void mneg(device const float* x,
                device float* out,
                uint index [[thread_position_in_grid]])
{
    out[index] = -1 * x[index];
}

kernel void mexp(device const float* x,
                 device float* out,
                 uint index [[thread_position_in_grid]])
{
    out[index] = exp(x[index]);
}

kernel void mlog(device const float* x,
                 device float* out,
                 uint index [[thread_position_in_grid]])
{
    out[index] = log(x[index]);
}