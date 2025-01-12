#include <stdlib.h>
#include "allocate.h"

float *memory(size_t nbytes)
{
    float *out = malloc(nbytes*sizeof(float));
    if (out == NULL) {
        return NULL;
    }

    return out;
}