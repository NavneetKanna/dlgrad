// same shape
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

// broadcast
kernel void add_arrays_2d(device const float* x,
                          device const float* y,
                          device float* out,
                          constant int* xshape,
                          constant int* xstrides,
                          constant int* yshape,
                          constant int* ystrides,
                          uint2 pos [[thread_position_in_grid]]) 
{
    int col = pos.x;
    int row = pos.y;

    int x_idx = row * xstrides[0] + col * xstrides[1];

    int y_idx1 = (xshape[0] == yshape[0]) ? row : 0;
    int y_idx2 = (xshape[1] == yshape[1]) ? col : 0;
    int y_idx = y_idx1 * ystrides[0] + y_idx2 * ystrides[1];

    out[x_idx] = x[x_idx] + y[y_idx];
}

kernel void add_arrays_3d(device const float* x,
                          device const float* y,
                          device float* out,
                          constant int* xshape,
                          constant int* xstrides,
                          constant int* yshape,
                          constant int* ystrides,
                          uint3 pos [[thread_position_in_grid]]) 
{
    int col = pos.x;
    int row = pos.y;
    int depth = pos.z;

    int x_idx = depth * xstrides[0] + row * xstrides[1] + col * xstrides[2];

    int y_idx1 = (xshape[0] == yshape[0]) ? depth : 0;
    int y_idx2 = (xshape[1] == yshape[1]) ? row : 0;
    int y_idx3 = (xshape[2] == yshape[2]) ? col : 0;
    int y_idx = y_idx1 * ystrides[0] + y_idx2 * ystrides[1] + y_idx3 * ystrides[2];

    out[x_idx] = x[x_idx] + y[y_idx];
}

kernel void sub_arrays_2d(device const float* x,
                          device const float* y,
                          device float* out,
                          constant int* xshape,
                          constant int* xstrides,
                          constant int* yshape,
                          constant int* ystrides,
                          uint2 pos [[thread_position_in_grid]]) 
{
    int col = pos.x;
    int row = pos.y;

    int x_idx = row * xstrides[0] + col * xstrides[1];

    int y_idx1 = (xshape[0] == yshape[0]) ? row : 0;
    int y_idx2 = (xshape[1] == yshape[1]) ? col : 0;
    int y_idx = y_idx1 * ystrides[0] + y_idx2 * ystrides[1];

    out[x_idx] = x[x_idx] - y[y_idx];
}

kernel void sub_arrays_3d(device const float* x,
                          device const float* y,
                          device float* out,
                          constant int* xshape,
                          constant int* xstrides,
                          constant int* yshape,
                          constant int* ystrides,
                          uint3 pos [[thread_position_in_grid]]) 
{
    int col = pos.x;
    int row = pos.y;
    int depth = pos.z;

    int x_idx = depth * xstrides[0] + row * xstrides[1] + col * xstrides[2];

    int y_idx1 = (xshape[0] == yshape[0]) ? depth : 0;
    int y_idx2 = (xshape[1] == yshape[1]) ? row : 0;
    int y_idx3 = (xshape[2] == yshape[2]) ? col : 0;
    int y_idx = y_idx1 * ystrides[0] + y_idx2 * ystrides[1] + y_idx3 * ystrides[2];

    out[x_idx] = x[x_idx] - y[y_idx];
}

kernel void mul_arrays_2d(device const float* x,
                          device const float* y,
                          device float* out,
                          constant int* xshape,
                          constant int* xstrides,
                          constant int* yshape,
                          constant int* ystrides,
                          uint2 pos [[thread_position_in_grid]]) 
{
    int col = pos.x;
    int row = pos.y;

    int x_idx = row * xstrides[0] + col * xstrides[1];

    int y_idx1 = (xshape[0] == yshape[0]) ? row : 0;
    int y_idx2 = (xshape[1] == yshape[1]) ? col : 0;
    int y_idx = y_idx1 * ystrides[0] + y_idx2 * ystrides[1];

    out[x_idx] = x[x_idx] * y[y_idx];
}

kernel void mul_arrays_3d(device const float* x,
                          device const float* y,
                          device float* out,
                          constant int* xshape,
                          constant int* xstrides,
                          constant int* yshape,
                          constant int* ystrides,
                          uint3 pos [[thread_position_in_grid]]) 
{
    int col = pos.x;
    int row = pos.y;
    int depth = pos.z;

    int x_idx = depth * xstrides[0] + row * xstrides[1] + col * xstrides[2];

    int y_idx1 = (xshape[0] == yshape[0]) ? depth : 0;
    int y_idx2 = (xshape[1] == yshape[1]) ? row : 0;
    int y_idx3 = (xshape[2] == yshape[2]) ? col : 0;
    int y_idx = y_idx1 * ystrides[0] + y_idx2 * ystrides[1] + y_idx3 * ystrides[2];

    out[x_idx] = x[x_idx] * y[y_idx];
}

kernel void div_arrays_2d(device const float* x,
                          device const float* y,
                          device float* out,
                          constant int* xshape,
                          constant int* xstrides,
                          constant int* yshape,
                          constant int* ystrides,
                          uint2 pos [[thread_position_in_grid]]) 
{
    int col = pos.x;
    int row = pos.y;

    int x_idx = row * xstrides[0] + col * xstrides[1];

    int y_idx1 = (xshape[0] == yshape[0]) ? row : 0;
    int y_idx2 = (xshape[1] == yshape[1]) ? col : 0;
    int y_idx = y_idx1 * ystrides[0] + y_idx2 * ystrides[1];

    out[x_idx] = x[x_idx] / y[y_idx];
}

kernel void div_arrays_3d(device const float* x,
                          device const float* y,
                          device float* out,
                          constant int* xshape,
                          constant int* xstrides,
                          constant int* yshape,
                          constant int* ystrides,
                          uint3 pos [[thread_position_in_grid]]) 
{
    int col = pos.x;
    int row = pos.y;
    int depth = pos.z;

    int x_idx = depth * xstrides[0] + row * xstrides[1] + col * xstrides[2];

    int y_idx1 = (xshape[0] == yshape[0]) ? depth : 0;
    int y_idx2 = (xshape[1] == yshape[1]) ? row : 0;
    int y_idx3 = (xshape[2] == yshape[2]) ? col : 0;
    int y_idx = y_idx1 * ystrides[0] + y_idx2 * ystrides[1] + y_idx3 * ystrides[2];

    out[x_idx] = x[x_idx] / y[y_idx];
}