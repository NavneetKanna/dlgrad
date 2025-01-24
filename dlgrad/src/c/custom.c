#include "stdlib.h"
#include "custom.h"
#include <stdio.h>


void ce_backward(float *x, float *target, int *xshape, int *xstride)
{
    int rows = xshape[0];
    int cols = xshape[1];

    for (int i=0; i<rows; i++) {
        x[(int)target[i]+(xstride[0]*i)] -= 1;
    }
}