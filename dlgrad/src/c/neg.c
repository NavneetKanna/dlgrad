#include <stdlib.h>
#include "neg.h"


void neg(float *x, float *out, int numel) {
    for (int i=0; i<numel; i++) {
        out[i] = -1 * x[i];
    }
}
