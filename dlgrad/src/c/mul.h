#ifndef MUL
#define MUL

float *mul_with_scalar(float *x, float *y, int xnumel);
float *mul_with_dim1(float *x, float *y, int xnumel, int at);
float *mul_with_dim0(float *x, float *y, int xnumel, int ynumel, int at);
float *mul(float *x, float *y, int xnumel);
float *mul_3d_with_2d(float *x, float *y, int xnumel, int ynumel);
float *mul_with_dim1_with_dim0(float *x, float *y, int xnumel, int ynumel, int at, int ncols);
void free_mul(float *ptr);
#endif