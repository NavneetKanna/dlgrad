#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "arithmetic.h"



float *op_3d(float *x, float *y, int xnumel, int ynumel, int *xshape, int *yshape, int *xstride, int *ystride, int yshape_len, int op) {
    float out_size = (xnumel >= ynumel) ? xnumel : ynumel;
    float *out = malloc(out_size * sizeof(float));

    int y_scalar = (yshape_len == 0);
    int y_row_or_1d = (yshape[0] == 1 || yshape_len == 1);
    int y_col = (yshape[1] == 1);
    int y_2d = (yshape_len == 2);
    int y_3d_dim1 = (yshape_len == 3 && yshape[0] == 1);

    for (int i=0; i<xshape[0]; i++) {
        for (int j=0; j<xshape[1]; j++) {
            for (int k=0; k<xshape[2]; k++) {
                int x_offset = i*xstride[0] + j*xstride[1] + k*xstride[2];
                int y_offset;

                if (y_scalar) {                                 // scalar
                    y_offset = 0;
                } else if (y_row_or_1d){                        // row Tensor or ndim=1 
                    y_offset = k*ystride[1];
                } else if (y_col) {                             // column Tensor
                    y_offset = j*ystride[0];
                } else if (y_2d) {                              // 2D Tensor
                    y_offset = j*ystride[0] + k*ystride[1]; 
                } else if (y_3d_dim1) {                         // 3D Tensor with first dim = 1
                    y_offset = j*ystride[0] + k*ystride[1];
                } else {                                        // 3D Tensor
                    y_offset = i*ystride[0] + j*ystride[1] + k*ystride[2];
                }

                switch(op) {
                    case 0: // Add
                        out[x_offset] = (xnumel >= ynumel) ? x[x_offset] + y[y_offset] : y[x_offset] + x[y_offset];
                        break;
                    case 1: // Subtract
                        out[x_offset] = (xnumel >= ynumel) ? x[x_offset] - y[y_offset] : x[y_offset] - y[x_offset];
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

float *op_2d(float *x, float *y, int xnumel, int ynumel, int *xshape, int *yshape, int *xstride, int *ystride, int yshape_len, int op) {
    float out_size = (xnumel >= ynumel) ? xnumel : ynumel;
    float *out = malloc(out_size * sizeof(float));
   
    int y_scalar = (yshape_len == 0);
    int y_row_or_1d = (yshape[0] == 1 || yshape_len == 1);
    int y_col = (yshape[1] == 1);

    for (int i=0; i<xshape[0]; i++) {
        for (int j=0; j<xshape[1]; j++) {
            int x_offset = i*xstride[0] + j*xstride[1];
            int y_offset = 0; 

            if (y_scalar) {                 // scalar
                y_offset = 0;
            } else if (y_row_or_1d) {       // row Tensor or ndim=1
                y_offset = j*ystride[1];
            } else if (y_col) {             // column Tensor
                y_offset = i*ystride[0];
            } else {                        // 2D Tensor
                y_offset = i*ystride[0] + j*ystride[1]; 
            }

            switch(op) {
                case 0: // Add
                    out[x_offset] = (xnumel >= ynumel) ? x[x_offset] + y[y_offset] : y[x_offset] + x[y_offset];
                    break;
                case 1: // Subtract
                    out[x_offset] = (xnumel >= ynumel) ? x[x_offset] - y[y_offset] : x[y_offset] - y[x_offset];
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