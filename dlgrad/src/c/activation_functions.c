#include <stdlib.h>
#include "activation_functions.h"


void relu(float *arr, float *out, int numel) {
    for (int i=0; i<numel; i++) {
        if (arr[i] <= 0) {
            out[i] = 0.0;
        } else {
            out[i] = arr[i];
        }
    }
}
