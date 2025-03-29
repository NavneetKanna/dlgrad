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
    out[index] = fast::exp(x[index]);
}

kernel void mlog(device const float* x,
                 device float* out,
                 uint index [[thread_position_in_grid]])
{
    out[index] = fast::log(x[index]);
}

kernel void mpow(device const float* x,
                 device float* out,
                 const device float* val,
                 uint index [[thread_position_in_grid]])
{
    out[index] = fast::pow(x[index], val[0]);
}

kernel void msqrt(device const float* x,
                 device float* out,
                 uint index [[thread_position_in_grid]])
{
    out[index] = fast::sqrt(x[index]);
}