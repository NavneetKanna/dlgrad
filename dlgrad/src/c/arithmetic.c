#include <stdlib.h>
#include <stdio.h>
#include "arithmetic.h"

#define ADD 0
#define MUL 1
#define SUB 2


float *add_with_scalar(float *x, float *y, int xnumel) 
{
    float *out = malloc(sizeof(float) * xnumel);

    for (int i=0; i<xnumel; i++) {
        out[i] = x[i] + y[0];
    }

    return out;
} 

float *add_with_dim1(float *x, float *y, int xnumel, int at)
{
    float *out = malloc(sizeof(float) * xnumel);

    int y_idx = 0;
    for (int i=0; i<xnumel; i++) {
        if (i!=0 && i%at==0)
            y_idx = 0; // At every 'at', set y_idx to 0
         
        out[i] = x[i] + y[y_idx];
        y_idx += 1;
    }

    return out;
}

float *add_with_dim0(float *x, float *y, int xnumel, int ynumel, int at)
{
    float *out = malloc(sizeof(float) * xnumel);

    int y_idx = 0;
    for (int i=0; i<xnumel; i++) {
        if (i!=0 && i%at==0) {
            y_idx += 1; // At every 'at', increment y_idx by 1
        }

        if (y_idx >= ynumel) 
            y_idx = 0;
        
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

float *add_with_dim1_with_dim0(float *x, float *y, int xnumel, int ynumel, int at, int ncols)
{
    float *out = malloc(sizeof(float) * xnumel);

    int start = 0;
    int y_idx = 0;
    for (int i=0; i<xnumel; i++) {
        if (i!=0 && i%at==0) {
            start += ncols;
            y_idx = start;
        }
        
        if (i!=0 && i%ncols==0) {
            y_idx = start;
        }
        
        out[i] = x[i] + y[y_idx];
        y_idx += 1;
    }

    return out;
}

void free_add(float *ptr) {
    free(ptr);
}