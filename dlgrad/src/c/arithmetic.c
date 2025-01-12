#include <stdlib.h>
#include <stdio.h>
#include "arithmetic.h"

#define ADD 0
#define MUL 1
#define SUB 2


// Handles all broadcasting shapes
void op_3d(float *x, float *y, float *out, int *xshape, int *xstrides, int *yshape, int *ystrides, int op) 
{
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
                y_idx3 = (xshape[2] == yshape[2]) ? k : 0;

                x_idx = i*xstrides[0] + j*xstrides[1] + k*xstrides[2];
                
                y_idx = y_idx1*ystrides[0] + y_idx2*ystrides[1] + y_idx3*ystrides[2];

                switch (op) {
                case ADD:
                    out[x_idx] = x[x_idx] + y[y_idx];
                    break;
                case MUL:
                    out[x_idx] = x[x_idx] * y[y_idx];
                    break;
                case SUB:
                    out[x_idx] = x[x_idx] - y[y_idx];
                    break;
                }
            }
        }
    }
}

// Handles all broadcasting shapes
float *op_2d(float *x, float *y, int *xshape, int *xstrides, int *yshape, int *ystrides, int outnumel, int op)
{
    float *out = malloc(outnumel*sizeof(float));

    int x_idx = 0;
    int y_idx = 0;
    int y_idx2 = 0;
    int y_idx1 = 0;

    for (int i=0; i<xshape[0]; i++) {
        y_idx1 = (xshape[0] == yshape[0]) ? i : 0;
        for (int j=0; j<xshape[1]; j++) {
            y_idx2 = (xshape[1] == yshape[1]) ? j : 0;

            x_idx = i*xstrides[0] + j*xstrides[1];
            
            y_idx = y_idx1*ystrides[0] + y_idx2*ystrides[1];

            switch (op) {
            case ADD:
                out[x_idx] = x[x_idx] + y[y_idx];
                break;
            case MUL:
                out[x_idx] = x[x_idx] * y[y_idx];
                break;
            case SUB:
                out[x_idx] = x[x_idx] - y[y_idx];
                break;
            }
        }
    }

    return out;
}

void free_op(float *ptr) {
    free(ptr);
}