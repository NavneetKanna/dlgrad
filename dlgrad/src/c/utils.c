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

void cpow(float *x, float *out, int val, int numel)
{
    for (int i=0; i<numel; i++) {
        out[i] = pow(x[i], val);
    }
}
