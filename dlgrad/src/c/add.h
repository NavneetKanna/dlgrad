#ifndef ADD
#define ADD

float *add_with_scalar(float *x, float *y, int xnumel);
float *add_with_dim1(float *x, float *y, int xnumel, int at);
float *add_with_dim0(float *x, float *y, int xnumel, int ynumel, int at);
float *add(float *x, float *y, int xnumel);
float *add_3d_with_2d(float *x, float *y, int xnumel, int ynumel);
float *add_with_dim1_with_dim0(float *x, float *y, int xnumel, int ynumel, int at, int ncols);
void free_add(float *ptr);
#endif