#include <stdlib.h>
#include "add.h"


float *add_with_scalar(float *x, float *y, int xnumel) 
{
    float *out = malloc(sizeof(float) * xnumel);

    for (int i=0; i<xnumel; i++) {
        out[i] = x[i] + y[0];
    }

    return out;
} 

float *add_with_dim1(float *x, float *y, int xnumel, int ncols)
{
    float *out = malloc(sizeof(float) * xnumel);

    int y_idx = 0;
    for (int i=0; i<xnumel; i++) {
        if (i!=0 && i%ncols==0)
            y_idx = 0; // At the start of new row, set y_idx to 0
         
        out[i] = x[i] + y[y_idx];
        y_idx += 1;
    }

    return out;
}

float *add_with_dim2(float *x, float *y, int xnumel, int ncols, int nrows)
{
    float *out = malloc(sizeof(float) * xnumel);

    int y_idx = 0;
    int n_rows_c = 0;
    for (int i=0; i<xnumel; i++) {
        if (i!=0 && i%ncols==0) {
            y_idx += 1; // At the start of new row, increment y_idx by 1
            n_rows_c += 1;
        }

        if (n_rows_c % nrows == 0)
            y_idx = 0; // For 3d case, at the start of new dim0, set y_idx to 0
        
        out[i] = x[i] + y[y_idx];
    }

    return out;
}

float *add(float *x, float *y, int xnumel)
{
    float *out = malloc(sizeof(float) * xnumel);

    for (int i=0; i<xnumel; i++)
        out[i] = x[i] + y[i];
    
    return out;
}

float *add_3d_with_2d(float *x, float *y, int xnumel, int ynumel)
{
    float *out = malloc(sizeof(float) * xnumel);
    
    int y_idx = 0;
    for (int i=0; i<xnumel; i++) {
        if (i!=0 && i%ynumel==0)
            y_idx = 0;
        
        out[i] = x[i] + y[y_idx];
        y_idx += 1;
    }

    return out;
}