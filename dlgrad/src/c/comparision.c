#include "comparision.h"
#include <stdlib.h>


float *gt_with_scalar(float *arr, float val, int numel)
{
    float *out = malloc(sizeof(float) * numel);

    for (int i=0; i<numel; i++) {
        if (arr[i] > val)
            out[i] = arr[i];
        else
            out[i] = 0.0;
    }

    return out;
}

void free_cmp(float *ptr)
{
    free(ptr);
}