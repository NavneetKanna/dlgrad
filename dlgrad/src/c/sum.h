#ifndef SUM
#define SUM

float *sum_3d_dim0(float *arr, int numel, int *shape, int *strides);
float *sum_3d_dim1(float *arr, int numel, int *shape, int *strides);
float *sum_3d_dim2(float *arr, int numel, int *shape, int *strides);
float *sum_2d_dim0(float *arr, int numel, int *shape, int *strides);
float *sum_2d_dim1(float *arr, int numel, int *shape, int *strides);

float *sum(float *x, int numel);
void free_sum(float *ptr);

#endif