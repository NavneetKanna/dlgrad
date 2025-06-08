#include <stdlib.h>
#include <math.h>
#include "utils.h"


void neg(float *x, float *out, int numel)
{
    for (int i=0; i<numel; i++) {
        out[i] = -1 * x[i];
    }
}

void cexp(float *x, float *out, int numel)
{
    for (int i=0; i<numel; i++) {
        out[i] = exp(x[i]);
    }
}

void clog(float *x, float *out, int numel)
{
    for (int i=0; i<numel; i++) {
        out[i] = log(x[i]);
    }
}

void cpow(float *x, float *out, float val, int numel)
{
    for (int i=0; i<numel; i++) {
        out[i] = pow(x[i], val);
    }
}

void csqrt(float *x, float *out, int numel)
{
    for (int i=0; i<numel; i++) {
        out[i] = sqrtf(x[i]);
    }
}

void argmax2d(float *x, float *out, int *xshape, int axis)
{
    int rows = xshape[0];
    int cols = xshape[1];
    if (axis == 1) {
        for (int i = 0; i < rows; i++) {
            float max = x[i * cols + 0];
            int idx = 0;
            for (int j = 1; j < cols; j++) {
                if (x[i * cols + j] > max) {
                    max = x[i * cols + j];
                    idx = j;
                }
            }
            out[i] = idx;
        }
    } else if (axis == 0) {
        for (int j = 0; j < cols; j++) {
            float max = x[0 * cols + j];
            int idx = 0;
            for (int i = 1; i < rows; i++) {
                if (x[i * cols + j] > max) {
                    max = x[i * cols + j];
                    idx = i;
                }
            }
            out[j] = idx;
        }
    } else {
        float max = -999;
        int idx = 0;
        for (int i=0; i<rows*cols; i++) {
            if (x[i] > max) {
                max = x[i];
                idx = i;
            }
        }

        out[0] = idx;
    }
}