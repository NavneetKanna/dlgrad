#ifndef SUM
#define SUM

float *sum_3d(float *x, int *xshape, int *xstride, int outnumel, int dim);
float *sum_2d(float *x, int *xshape, int *xstride, int outnumel, int dim);
float *sum(float *x, int numel);
void free_sum(float *ptr);

#endif