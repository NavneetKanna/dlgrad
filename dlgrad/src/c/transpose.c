#include <stdlib.h>
#include "transpose.h"


// only 2d
void transpose(float *x, float *out, int xrows, int xcols, int *xstride, int *outstride)
{
    for (int i=0; i<xrows; i++) {
        for (int j=0; j<xcols; j++) {
            int out_idx = j*outstride[0] + i;
            int x_idx = i*xstride[0] + j*xstride[1];
            out[out_idx] = x[x_idx];
        }
    }
}