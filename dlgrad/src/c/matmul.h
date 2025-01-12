#ifndef MATMUL
#define MATMUL

void matmul(float *x, float *y, float *out, int x_rows, int y_cols, int y_rows, int *ystride, int *xstride);

#endif