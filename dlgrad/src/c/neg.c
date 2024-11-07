#include <stdlib.h>
#include "neg.h"


float *neg(float *x, int numel) {
    float *out = malloc(numel * sizeof(float));

    for (int i=0; i<numel; i++) {
        out[i] = -1 * x[i];
    }

    return out;
}

void free_neg(float *ptr){
    free(ptr);
}