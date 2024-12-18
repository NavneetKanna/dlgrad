#include <stdlib.h>
#include "sum.h"


float *sum_3d_dim0(float *arr, int numel, int dim0, int dim1, int dim2, int *strides) {
    float *out = malloc(sizeof(float)*numel);

    for (int i=0; i<dim1; i++) { // rows
        for (int j=0; j<dim2; j++) { // cols
            float sum;
            for (int k=0; k<dim0; k++) { 
                sum += arr[j*strides[2] + i*strides[1] + k*strides[0]];
            }
            out[i*strides[1] + j*strides[2]] = sum;
        }
    }
}

float *sum_3d_dim1(float *arr, int numel, int *shape, int *strides) {
    float *out = malloc(sizeof(float)*numel);

    for(int i=0; i<shape[0]; i+=strides[0]) {
        for (int j=0; j<shape[2]; j++) { // cols
            float sum = 0.0;
            for(int k=i; k<(i+=strides[0]); k+=strides[1]) { // rows
                sum += arr[k];
            }
            out[i] = sum;
        }
    }
}


float *sum(float *x, int numel) {
    float *out = malloc(1 * sizeof(float));

    float sum = 0.0;
    for (int i=0; i<numel; i++) {
        sum += x[i];
    }

    out[0] = sum;

    return out;
}

void free_sum(float *ptr) {
    free(ptr);
}