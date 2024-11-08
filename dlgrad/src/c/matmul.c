#include <stdlib.h>
#include "matmul.h"


// TODO: Naive solution, will optimise later
float *matmul(float *x, float *y, int x_rows, int y_cols, int y_rows) {
    float *out = malloc(x_rows*y_cols*sizeof(float));
    for (int i = 0; i < x_rows * y_cols; i++) {
        out[i] = 0.0f;
    }

    for (int i=0; i<x_rows; i++) {
        for (int j=0; j<y_cols; j++) {
            for (int k=0; k<y_rows; k++) {
                out[i*y_cols + j] += x[i*y_rows + k] * y[k*y_cols + j];
            }
        }
    }

    return out;
}

void free_matmul(float *ptr) {
    free(ptr);
}