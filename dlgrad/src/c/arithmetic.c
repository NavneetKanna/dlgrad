#include <stdlib.h>
#include <stdio.h>
#include "arithmetic.h"

#define ADD 0
#define MUL 1
#define SUB 2


// Handles all broadcasting shapes
float *add(float *x, float *y, int *xshape, int *xstrides, int *yshape, int *ystrides) 
{
    float *out = malloc(sizeof(float));

    int x_idx = 0;
    int y_idx = 0;
    int y_idx3 = 0;
    int y_idx2 = 0;
    int y_idx1 = 0;

    for (int i=0; i<xshape[0]; i++) {
        y_idx1 = (xshape[0] == yshape[0]) ? i : 0;
       
        for (int j=0; j<xshape[1]; j++) {
            y_idx2 = (xshape[1] == yshape[1]) ? j : 0;

            for (int k=0; k<xshape[2]; k++) {
                x_idx = i*xstrides[0] + j*xstrides[1] + k*xstrides[2];
                y_idx3 = (xshape[2] == yshape[2]) ? k : 0;
                
                y_idx = y_idx1*ystrides[0] + y_idx2*ystrides[1] + y_idx3*ystrides[2];

                out[x_idx] = x[x_idx] + y[y_idx];
            }
        }
    }

    return out;
}


void free_add(float *ptr) {
    free(ptr);
}