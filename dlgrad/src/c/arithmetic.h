#ifndef ARITHMETIC
#define ARITHMETIC

void op_3d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op);
void op_2d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op);
void with_scalar(float *x, float *out, float *y, int xnumel, int op);

#endif 