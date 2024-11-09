#include <stdlib.h>
#include <stdio.h>
#include "matmul.h"


// TODO: Naive solution, will optimise later
float *matmul(float *x, float *y, int x_rows, int y_cols, int y_rows, int *ystride, int *xstride) {
    float *out = malloc(x_rows*y_cols*sizeof(float));
    // for (int i = 0; i < x_rows * y_cols; i++) {
    //     out[i] = 0.0f;
    // }

    float sum = 0.0;
    for (int i=0; i<x_rows; i++) {
        for (int j=0; j<y_cols; j++) {
            sum = 0.0;
            for (int k=0; k<y_rows; k++) {
                // out[i*y_cols + j] += x[i*y_rows + k] * y[k*y_cols + j];
                sum += x[i*xstride[0] + k*xstride[1]] * y[k*ystride[0] + j*ystride[1]];
                // printf("%d %f\n", k*ystride[0] + j*ystride[1], y[k*ystride[0] + j*ystride[1]]);
            }
            out[i*y_cols + j] = sum;
        }
    }

    return out;
}

void free_matmul(float *ptr) {
    free(ptr);
}