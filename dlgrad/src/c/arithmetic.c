#include <stdlib.h>
#include <stdio.h>
#include "arithmetic.h"

#define ADD 0
#define MUL 1
#define SUB 2


void add(float *x, float *y, int *xshape, int *xstrides, int *yshape, int *ystrides) 
{
    int y_idx3 = 0;
    int y_idx2 = 0;
    int y_idx1 = 0;
    for (int i=0; i<xshape[0]; i++) {
        if (xshape[0] == yshape[0]) {
                y_idx1 = i;
            } else {
                y_idx1 = 0;
            }
        for (int j=0; j<xshape[1]; j++) {
            if (xshape[1] == yshape[1]) {
                y_idx2 = j;
            } else {
                y_idx2 = 0;
            }
            for (int k=0; k<xshape[2]; k++) {
                int x_idx = i*xstrides[0] + j*xstrides[1] + k*xstrides[2];
                if (xshape[2] == yshape[2]) {
                    y_idx3 = k;
                } else {
                    y_idx3 = 0;
                }
                
                int y_idx = y_idx1*ystrides[0] + y_idx2*ystrides[1] + y_idx3*ystrides[2];
            }
        }
    }
    
}


void free_add(float *ptr) {
    free(ptr);
}