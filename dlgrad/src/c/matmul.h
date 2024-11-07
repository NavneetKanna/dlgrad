#ifndef MATMUL
#define MATMUL

float *matmul(float *x, float *y, int x_rows, int y_cols, int y_rows);
void free_matmul(float *ptr);

#endif