#ifndef MAX
#define MAX

#include <stdbool.h>

void max_3d(float *x, float *out, float *tmp, float *maxs_with_1s, int *xshape, int *xstride, int outnumel, int dim);
void mmax_2d(float *x, float *out, int *xshape, int *xstride, int outnumel, int dim);
void max_2d(float *x, float *out, float *tmp, float *maxs_with_1s, int *xshape, int *xstride, int outnumel, int dim);
void max(float *x, float *out, int numel);
void new_max(float *x, float *out, int *stride, int *shape, int numel, int dim);

#endif