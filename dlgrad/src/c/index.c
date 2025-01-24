#include <stdlib.h>
#include "index.h"


// only for nll loss
void indexing(float *x, float *out, int *xshape, int *xstride, float *idxs)
{
    for (int i=0; i<xshape[0]; i++) {   // rows
        int out_idx = idxs[i]*xstride[1] + i*xstride[0];
        out[i] = x[out_idx];
    }
}