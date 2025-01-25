#include <stdlib.h>
#include "full.h"


void full(float *out, int numel, float fill_value) 
{
    for (int i=0; i<numel; i++) {
        out[i] = fill_value;
    }
}
