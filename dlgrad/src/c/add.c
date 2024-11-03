#include <stdlib.h>
#include <stdio.h>
#include "add.h"


// TODO: Optimise

int get_y_idx_3d(int dim1, int dim2, int dim3, int *yshape, int *ystride) {
    int idx = 0;

    if (yshape[0] != 1) {
        idx += dim1*ystride[0];
    } 

    if (yshape[1] != 1) {
        idx += dim2*ystride[1];
    }

    if (yshape[2] != 1) {
        idx += dim3*ystride[2];
    }

    return idx;
}

float *add_3d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride) {
    float *out = malloc(numel * sizeof(float));

    for (int i=0; i<xshape[0]; i++) {
        for (int j=0; j<xshape[1]; j++) {
            for (int k=0; k<xshape[2]; k++) {
                int offset = i*xstride[0] + j*xstride[1] + k*xstride[2];
                out[offset] = x[offset] + y[get_y_idx_3d(i, j, k, yshape, ystride)];
            }
        }
    }

    return out;
}

int get_y_idx_2d(int dim1, int dim2, int *yshape, int *ystride) {
    int idx = 0;

    if (yshape[0] != 1) {
        idx += dim1*ystride[0];
    } 

    if (yshape[1] != 1) {
        idx += dim2*ystride[1];
    }

    return idx;
}

float *add_2d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride) {
    // printf("in add 2d\n");
    // printf("numel: %d\n", numel);
    // printf("xshape[0] %d\n", xshape[0]);
    // printf("xshape[1] %d\n", xshape[1]);
    // printf("xstride[0] %d\n", xstride[0]);
    // printf("xstride[1] %d\n", xstride[1]);
    // printf("yshape[0] %d\n", yshape[0]);
    // printf("yshape[1] %d\n", yshape[1]);
    // printf("ystride[0] %d\n", ystride[0]);
    // printf("ystride[1] %d\n", ystride[1]);

    float *out = malloc(numel * sizeof(float));

    for (int i=0; i<xshape[0]; i++) {
        // printf("\n----\n");
        for (int j=0; j<xshape[1]; j++) {
            int offset = i*xstride[0] + j*xstride[1]; 
            // printf("y idx %d ", get_y_idx_2d(i, j, yshape, ystride));

            out[offset] = x[offset] + y[get_y_idx_2d(i, j, yshape, ystride)];
        }
    }

    return out;
}

// see my blog for the explanation 
// https://navneetkanna.github.io/blog/2024/02/22/dlgrad-Behind-the-scenes.html
float *add(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride, int ndim) {
    switch (ndim)
    {
    case 2:
        return add_2d(x, y, numel, xshape, yshape, xstride, ystride);
        break;
    case 3:
        return add_3d(x, y, numel, xshape, yshape, xstride, ystride);
        break;
    
    default:
        break;
    }
}

void free_add(float* ptr) {
    free(ptr);
}