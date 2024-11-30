#ifndef ARITHMETIC
#define ARITHMETIC

float *op_2d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride, int yshape_len, int op);
float *op_3d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride, int yshape_len, int op);
void free_op(float *ptr);

#endif 