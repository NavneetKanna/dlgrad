#include <stdlib.h>
#include "loss.h"

void ce_forward(float *x, float *target, float *out, int nrows, int *xstride)
{
    for (int i=0; i<nrows; i++) {
        out[i] = x[(int)target[i]+(xstride[0]*i)];
    }
}

void ce_backward(float *x, float *target, int *xshape, int *xstride)
{
    int rows = xshape[0];
    int cols = xshape[1];

    for (int i=0; i<rows; i++) {
        x[(int)target[i]+(xstride[0]*i)] -= 1;
    }
}