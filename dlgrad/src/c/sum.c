#include <stdlib.h>
#include "sum.h"

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