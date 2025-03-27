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


kernel void add_arrays_2d(device const float* x,
                          device const float* y,
                          device float* out,
                          constant int* xshape,    // Shape of x: [rows, cols]
                          constant int* xstrides, // Strides of x: [stride0, stride1]
                          constant int* yshape,    // Shape of y: [rows, cols]
                          constant int* ystrides, // Strides of y: [stride0, stride1]
                          uint2 pos [[thread_position_in_grid]]) 
{
    int col = pos.x;
    int row = pos.y;

    // int x_idx = pos.y * xstrides[0] + pos.x * xstrides[1];
    int x_idx = row * xstrides[0] + col * xstrides[1];

    int y_idx1 = (xshape[0] == yshape[0]) ? row : 0; // Row index (broadcast if yshape[0] == 1)
    int y_idx2 = (xshape[1] == yshape[1]) ? col : 0; // Col index (broadcast if yshape[1] == 1)
    int y_idx = y_idx1 * ystrides[0] + y_idx2 * ystrides[1];

    out[x_idx] = x[x_idx] + y[y_idx];
}

kernel void add_arrays_3d(device const float* x,
                          device const float* y,
                          device float* out,
                          constant int* xshape,    // Shape of x: [depth, rows, cols]
                          constant int* xstrides, // Strides of x: [stride0, stride1, stride2]
                          constant int* yshape,    // Shape of y: [depth, rows, cols]
                          constant int* ystrides, // Strides of y: [stride0, stride1, stride2]
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