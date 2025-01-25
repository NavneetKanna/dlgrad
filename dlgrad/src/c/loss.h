#ifndef LOSS
#define LOSS

void ce_forward(float *x, float *target, float *out, int nrows, int *xstride);
void ce_backward(float *x, float *target, int *xshape, int *xstride);

#endif