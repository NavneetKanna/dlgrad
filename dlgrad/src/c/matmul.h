#ifndef MATMUL
#define MATMUL

float *matmul(float *x, float *y, int x_rows, int y_cols, int y_rows, int *ystride, int *xstride);
void free_matmul(float *ptr);

#endif