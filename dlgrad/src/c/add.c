#include <stdlib.h>
#include "add.h"


float *add_2d_scalar(float *x, float *y, int xnumel) 
{
    float *out = malloc(sizeof(float) * xnumel);

    for (int i=0; i<xnumel; i++) {
        out[i] = x[i] + y[0];
    }

    return out;
} 

float *add_2d_dim1(float *x, float *y, int xnumel, int ncols)
{
    float *out = malloc(sizeof(float) * xnumel);

    int y_idx = 0;
    for (int i=0; i<xnumel; i++) {
        if (i!=0 && i%ncols==0)
            y_idx = 0; // At the start of new row, set y_idx to 0
         
        out[i] = x[i] + y[y_idx];
        y_idx += 1;
    }
}

float *add_2d_dim2(float *x, float *y, int xnumel, int ncols)
{
    float *out = malloc(sizeof(float) * xnumel);

    int y_idx = 0;
    for (int i=0; i<xnumel; i++) {
        if (i!=0 && i%ncols==0)
            y_idx += 1; // At the start of new row, increment y_idx by 1
        
        out[i] = x[i] + y[y_idx];
    }

    return out;
}