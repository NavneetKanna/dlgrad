#include "comparision.h"
#include <stdlib.h>


void gt_with_scalar(float *arr, float *out, float val, int numel)
{
    for (int i=0; i<numel; i++) {
        if (arr[i] > val)
            out[i] = 1.0;
        else
            out[i] = 0.0;
    }
}
