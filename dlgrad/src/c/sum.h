#ifndef SUM
#define SUM

void sum_3d(float *x, float *out, int *xshape, int *xstride, int outnumel, int dim);
void sum_2d(float *x, float *out, int *xshape, int *xstride, int outnumel, int dim);
void sum(float *x, float *out, int numel);

#endif