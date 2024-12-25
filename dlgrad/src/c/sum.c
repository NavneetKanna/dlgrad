#include <stdlib.h>
#include <stdio.h>
#include "sum.h"


float *sum_3d_dim0(float *arr, int numel, int *shape, int *strides) {
    float *out = malloc(sizeof(float)*numel);
    int idx = 0;

    for (int i=0; i<shape[1]; i++) { // rows
        for (int j=0; j<shape[2]; j++) { // cols
            float sum = 0.0;
            for (int k = 0; k < shape[0]; k++) {
                sum += arr[k * strides[0] + i * strides[1] + j * strides[2]];
            }
            out[idx++] = sum;
        }
    }

    return out;
}

float *sum_3d_dim1(float *arr, int numel, int *shape, int *strides) {
    float *out = malloc(sizeof(float)*numel);
    int idx = 0;

    for(int i=0; i<shape[0]*strides[0]; i+=strides[0]) {
        for (int j=0; j<shape[2]; j++) { // cols
            float sum = 0.0;
            for(int k=i+j; k<(i+strides[0]); k+=strides[1]) { // rows
                sum += arr[k];
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
    
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) { // rows
            float sum = 0.0;
            for (int k = 0; k < shape[2]; k++) { // cols
                sum += arr[i * strides[0] + j * strides[1] + k * strides[2]];
            }
            out[idx++] = sum;
        }
    }

    return out;
}

float *sum_2d_dim0(float *arr, int numel, int *shape, int *strides) {
    float *out = malloc(sizeof(float)*numel);
    int idx = 0;

    for (int i=0; i<shape[1]; i++) { // cols
        float sum = 0.0;
        for (int j=0; j<shape[0]; j++) { // rows
            sum += arr[i + j*strides[0]];
        }
        out[idx++] = sum;
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