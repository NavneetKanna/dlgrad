kernel void add_arrays(device const float* x,
                       device const float* y,
                       device float* out,
                       uint index [[thread_position_in_grid]])
{
    out[index] = x[index] + y[index];
}

kernel void sub_arrays(device const float* x,
                       device const float* y,
                       device float* out,
                       uint index [[thread_position_in_grid]])
{
    out[index] = x[index] - y[index];
}

kernel void mul_arrays(device const float* x,
                       device const float* y,
                       device float* out,
                       uint index [[thread_position_in_grid]])
{
    out[index] = x[index] * y[index];
}

kernel void div_arrays(device const float* x,
                       device const float* y,
                       device float* out,
                       uint index [[thread_position_in_grid]])
{
    out[index] = x[index] / y[index];
}
