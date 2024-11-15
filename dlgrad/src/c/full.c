#include <stdlib.h>
#include "full.h"


float *full(int numel, float fill_value) {
    float *out = malloc(numel * sizeof(float));

    for (int i=0; i<numel; i++) {
        out[i] = fill_value;
    }

    return out;
}

void *free_full(float *ptr) {
    free(ptr);
}