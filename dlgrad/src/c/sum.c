#include <stdlib.h>
#include "sum.h"


float *sum_3d_dim0(float *arr, int numel, int *shape, int *strides) {
    float *out = malloc(sizeof(float)*numel);
    int idx = 0;
    for (int i=0; i<shape[1]; i++) { // rows
        for (int j=0; j<shape[2]; j++) { // cols
            float sum = 0.0;
            for (int k=j; k<shape[0]; k+=strides[0]) {
                sum += arr[k];
            }
            out[idx] = sum;
            idx += 1;
        }
    }

    return out;
}

float *sum_3d_dim1(float *arr, int numel, int *shape, int *strides) {
    float *out = malloc(sizeof(float)*numel);
    int idx = 0;
    for(int i=0; i<shape[0]; i+=strides[0]) {
        for (int j=0; j<shape[2]; j++) { // cols
            float sum = 0.0;
            for(int k=i; k<(i+=strides[0]); k+=strides[1]) { // rows
                sum += arr[k+i];
            }
            out[idx] = sum;
            idx += 1;
        }
    }

    return out;
}

float *sum_3d_dim2(float *arr, int numel, int *shape, int *strides) {
    float *out = malloc(sizeof(float)*numel);
    int idx = 0;
    for(int i=0; i<shape[0]; i+=strides[0]) {
        for (int j=0; j<shape[1]; j++) { // rows
            float sum = 0.0;
            for (int k=0; k<shape[2]; k++) { // cols
                sum += arr[k+i];
            }
            out[idx] = sum;
            idx += 1;
        }
    }

    return out;
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