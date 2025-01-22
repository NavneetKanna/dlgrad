#ifndef MAX
#define MAX

void max_3d(float *x, float *out, float *tmp, float *maxs_with_1s, int *xshape, int *xstride, int outnumel, int dim);
void max_2d(float *x, float *out, float *tmp, float *maxs_with_1s, int *xshape, int *xstride, int outnumel, int dim);
void max(float *x, float *out, int numel);

#endif