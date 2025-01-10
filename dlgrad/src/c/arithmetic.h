#ifndef ARITHMETIC
#define ARITHMETIC

float *op_3d(float *x, float *y, int *xshape, int *xstrides, int *yshape, int *ystrides, int outnumel, int op);
float *op_2d(float *x, float *y, int *xshape, int *xstrides, int *yshape, int *ystrides, int outnumel, int op);
void free_add(float *ptr);

#endif 