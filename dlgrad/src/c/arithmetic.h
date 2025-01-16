#ifndef ARITHMETIC
#define ARITHMETIC

void op_3d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op);
void op_2d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op);
void add_with_1d(float *x, float *y, float *out, int xnumel, int ynumel, int op);

#endif 