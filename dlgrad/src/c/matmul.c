#include <stdlib.h>
#include "matmul.h"


// TODO: Naive solution, will optimise later
void matmul(float *x, float *y, float *out, int x_rows, int y_cols, int y_rows, int *ystride, int *xstride) {
    float sum = 0.0;
    for (int i=0; i<x_rows; i++) {
        for (int j=0; j<y_cols; j++) {
            sum = 0.0;
            for (int k=0; k<y_rows; k++) {
                sum += x[i*xstride[0] + k*xstride[1]] * y[k*ystride[0] + j*ystride[1]];
            }
            out[i*y_cols + j] = sum;
        }
    }
}
