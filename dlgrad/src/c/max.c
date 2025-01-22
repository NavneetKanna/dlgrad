#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "max.h"



void max(float *x, float *out, int numel) {
    float max = 0.0;
    for (int i=0; i<numel; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    out[0] = max;
}

void max_3d(float *x, float *out, float *tmp, float *maxs_with_1s, int *xshape, int *xstride, int outnumel, int dim)
{
    if (dim == -1) {                // elementwise
        return max(x, out, outnumel);
    }

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
                        break;
                    case 1:
                        out_idx = od*ncols + col;
                        break;
                    case 2:
                        out_idx = od*nrows + row;
                        break;
                }
                if (x[x_idx] > out[out_idx]) {
                    out[out_idx] = x[x_idx];
                }
            }
        }
    }

    switch (dim)
    {
    case 0:
        for (int i=0; i<nrows*ncols; i++) {
            maxs_with_1s[(int)tmp[i]] = 1.0f;
        }
        break;
    case 1:
        for (int i=0; i<nouter_dim*ncols; i++) {
            maxs_with_1s[(int)tmp[i]] = 1.0f;
        }
        break;
    case 2:
        for (int i=0; i<nouter_dim*nrows; i++) {
            maxs_with_1s[(int)tmp[i]] = 1.0f;
        }
        break;
    }
}

void max_2d(float *x, float *out, float *tmp, float *maxs_with_1s, int *xshape, int *xstride, int outnumel, int dim)
{
    if (dim == -1) {                // elementwise 
        return max(x, out, outnumel);
    }

    int out_idx = 0;
    int x_idx = 0;
    int tmp_max = 0;
    
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
                    break;
                case 1:
                    out_idx = row;
                    break;
            }
            if (x[x_idx] > out[out_idx]) {
                out[out_idx] = x[x_idx];
                tmp[out_idx] = x_idx;
            }
        }
    }

    switch (dim)
    {
    case 0:
        for (int i=0; i<ncols; i++) {
            maxs_with_1s[(int)tmp[i]] = 1.0f;
        }
        break;
    case 1:
        for (int i=0; i<nrows; i++) {
            maxs_with_1s[(int)tmp[i]] = 1.0f;
        }
        break;
    }
}
