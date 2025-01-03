#ifndef SUB
#define SUB

float *sub_with_scalar(float *x, float *y, int xnumel);
float *sub_with_dim1(float *x, float *y, int xnumel, int at);
float *sub_with_dim0(float *x, float *y, int xnumel, int ynumel, int at);
float *sub(float *x, float *y, int xnumel);
float *sub_3d_with_2d(float *x, float *y, int xnumel, int ynumel);
float *sub_with_dim1_with_dim0(float *x, float *y, int xnumel, int ynumel, int at, int ncols);
void free_sub(float *ptr);

#endif