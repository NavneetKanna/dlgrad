#include <metal_math>
using namespace metal;



kernel void mneg2d(device const float* x, device float* out, constant int* xstrides, uint2 pos [[thread_position_in_grid]])
{
    int idx = pos.y * xstrides[0] + pos.x * xstrides[1];
    out[idx] = -1 * x[idx];
}

kernel void mneg3d(device const float* x, device float* out, constant int* xstrides, uint3 pos [[thread_position_in_grid]])
{
    int idx = pos.z * xstrides[0] + pos.y * xstrides[1] + pos.x * xstrides[2];
    out[idx] = -1 * x[idx];
}



kernel void mexp2d(device const float* x, device float* out, constant int* xstrides, uint2 pos [[thread_position_in_grid]])
{
    int idx = pos.y * xstrides[0] + pos.x * xstrides[1];
    out[idx] = fast::exp(x[idx]);
}

kernel void mexp3d(device const float* x, device float* out, constant int* xstrides, uint3 pos [[thread_position_in_grid]]) 
{
    int idx = pos.z * xstrides[0] + pos.y * xstrides[1] + pos.x * xstrides[2];
    out[idx] = fast::exp(x[idx]);
}



kernel void mlog2d(device const float* x, device float* out, constant int* xstrides, uint2 pos [[thread_position_in_grid]])
{
    int idx = pos.y * xstrides[0] + pos.x * xstrides[1];
    out[idx] = fast::log(x[idx]);
}

kernel void mlog3d(device const float* x, device float* out, constant int* xstrides, uint3 pos [[thread_position_in_grid]]) 
{
    int idx = pos.z * xstrides[0] + pos.y * xstrides[1] + pos.x * xstrides[2];
    out[idx] = fast::log(x[idx]);
}



kernel void mpow2d(device const float* x, device float* out, const device float* val, constant int* xstrides, uint2 pos [[thread_position_in_grid]])
{
    int idx = pos.y * xstrides[0] + pos.x * xstrides[1];
    out[idx] = fast::pow(x[idx], val[0]);
}

kernel void mpow3d(device const float* x, device float* out, const device float* val, constant int* xstrides, uint3 pos [[thread_position_in_grid]]) 
{
    int idx = pos.z * xstrides[0] + pos.y * xstrides[1] + pos.x * xstrides[2];
    out[idx] = fast::pow(x[idx], val[0]);
}



kernel void msqrt2d(device const float* x, device float* out, constant int* xstrides, uint2 pos [[thread_position_in_grid]])
{
    int idx = pos.y * xstrides[0] + pos.x * xstrides[1];
    out[idx] = fast::sqrt(x[idx]);
}

kernel void msqrt3d(device const float* x, device float* out, constant int* xstrides, uint3 pos [[thread_position_in_grid]]) 
{
    int idx = pos.z * xstrides[0] + pos.y * xstrides[1] + pos.x * xstrides[2];
    out[idx] = fast::sqrt(x[idx]);
}
