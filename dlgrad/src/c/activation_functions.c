#include <stdlib.h>
#include "activation_functions.h"


float *relu(float *arr, int numel) {
    float *out = malloc(numel * sizeof(float));

    for (int i=0; i<numel; i++) {
        if (arr[i] <= 0) {
            out[i] = 0.0;
        } else {
            out[i] = arr[i];
        }
    }

    return out;
}

void free_af(float *ptr) {
    free(ptr);
}