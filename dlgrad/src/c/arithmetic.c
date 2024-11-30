#include <stdlib.h>
#include "arithmetic.h"


// Assumptions: x is the "bigger" Tensor

float *op_3d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride, int yshape_len, int op) {
    float *out = malloc(numel * sizeof(float));

    for (int i=0; i<xshape[0]; i++) {
        for (int j=0; j<xshape[1]; j++) {
            for (int k=0; k<xshape[2]; k++) {
                int x_offset = i*xstride[0] + j*xstride[1] + k*xstride[2];
                int y_offset;

                if (yshape_len == 0) { // scalar
                    y_offset = 0;
                } else if (yshape[0] == 1 || yshape_len == 1){ // row Tensor or ndim=1 
                    y_offset = k*ystride[1];
                } else if (yshape[1] == 1) { // column Tensor
                    y_offset = j*ystride[0];
                } else if (yshape_len == 2) { // 2D Tensor
                    y_offset = j*ystride[0] + k*ystride[1]; 
                } else if (yshape_len == 3 && yshape[0] == 1) { // 3D Tensor with first dim = 1
                    y_offset = j*ystride[0] + k*ystride[1];
                } else { // 3D Tensor
                    y_offset = i*ystride[0] + j*ystride[1] + k*ystride[2];
                }

                switch(op) {
                    case 0: // Add
                        out[x_offset] = x[x_offset] + y[y_offset];
                        break;
                    case 1: // Subtract
                        out[x_offset] = x[x_offset] - y[y_offset];
                        break;
                    case 2: // Multiply
                        out[x_offset] = x[x_offset] * y[y_offset];
                        break;
                    case 3: // Divide
                        out[x_offset] = x[x_offset] / y[y_offset];
                        break;
                }
            }
        }
    }

    return out;
}

float *op_2d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride, int yshape_len, int op) {
    float *out = malloc(numel * sizeof(float));

    for (int i=0; i<xshape[0]; i++) {
        for (int j=0; j<xshape[1]; j++) {
            int x_offset = i*xstride[0] + j*xstride[1];
            int y_offset = 0; 
            
            if (yshape_len == 0) { // scalar
                y_offset = 0;
            } else if (yshape[0] == 1 || yshape_len == 1) { // row Tensor or ndim=1
                y_offset = j*ystride[1];
            } else if (yshape[1] == 1) { // column Tensor
                y_offset = i*ystride[0];
            } else { // 2D Tensor
                y_offset = i*ystride[0] + j*ystride[1]; 
            }

            switch(op) {
                case 0: // Add
                    out[x_offset] = x[x_offset] + y[y_offset];
                    break;
                case 1: // Subtract
                    out[x_offset] = x[x_offset] - y[y_offset];
                    break;
                case 2: // Multiply
                    out[x_offset] = x[x_offset] * y[y_offset];
                    break;
                case 3: // Divide
                    out[x_offset] = x[x_offset] / y[y_offset];
                    break;
            }
        }
    }

    return out;
}

void free_op(float* ptr) {
    free(ptr);
}