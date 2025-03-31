kernel void sum2d(device const float* x,
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

}