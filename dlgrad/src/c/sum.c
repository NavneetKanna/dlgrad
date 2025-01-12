#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "sum.h"


// TODO: Optimise by reording loops for each dim ?
// TODO: Check if memset is efficient or calloc
// TODO: Use macros for dim ?

float *sum(float *x, int numel) {
    float *out = malloc(1 * sizeof(float));

    float sum = 0.0;
    for (int i=0; i<numel; i++) {
        sum += x[i];
    }

    out[0] = sum;

    return out;
}

float *sum_3d(float *x, int *xshape, int *xstride, int outnumel, int dim)
{
    if (dim == -1) {                // elementwise
        return sum(x, outnumel);
    }

    float *out = malloc(outnumel * sizeof(float));
    memset(out, 0, outnumel * sizeof(float));
    
    int out_idx = 0;
    int x_idx = 0;
    
    int nouter_dim = xshape[0];
    int nrows = xshape[1];
    int ncols = xshape[2];
    
    int od_stride = xstride[0];
    int row_stride = xstride[1];
    int col_stride = xstride[2];
    
    for (int od=0; od<nouter_dim; od++) {
        for (int row=0; row<nrows; row++) {
            for (int col=0; col<ncols; col++) {
                x_idx = od*od_stride + row*row_stride + col*col_stride;
                switch (dim) {
                    case 0:
                        out_idx = row*ncols + col;
                        out[out_idx] += x[x_idx];
                        break;
                    case 1:
                        out_idx = od*ncols + col;
                        out[out_idx] += x[x_idx];
                        break;
                    case 2:
                        out_idx = od*nrows + row;
                        out[out_idx] += x[x_idx];
                        break;
                    case -1:
                        break;
                }
            }
        }
    }
    
    return out;
}

float *sum_2d(float *x, int *xshape, int *xstride, int outnumel, int dim)
{
    if (dim == -1) {                // elementwise 
        return sum(x, outnumel);
    }

    float *out = malloc(outnumel * sizeof(float));
    memset(out, 0, outnumel * sizeof(float));
    
    int out_idx = 0;
    int x_idx = 0;
    
    int nrows = xshape[0];
    int ncols = xshape[1];
    
    int row_stride = xstride[0];
    int col_stride = xstride[1];
    
    for (int row=0; row<nrows; row++) {
        for (int col=0; col<ncols; col++) {
            x_idx = row*row_stride + col*col_stride;
            switch (dim) {
                case 0:
                    out_idx = col;
                    out[out_idx] += x[x_idx];
                    break;
                case 1:
                    out_idx = row;
                    out[out_idx] += x[x_idx];
                    break;
            }
        }
    }
    
    return out;
}

void free_sum(float *ptr) {
    free(ptr);
}