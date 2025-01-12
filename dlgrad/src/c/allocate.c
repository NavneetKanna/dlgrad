#include <stdlib.h>
#include "allocate.h"


float *uninitialized_memory(size_t num)
{
    float *out = malloc(num*sizeof(float));
    if (out == NULL) {
        return NULL;
    }

    return out;
}

float *initialized_memory(size_t num)
{
    float *out = calloc(num, sizeof(float));
    if (out == NULL) {
        return NULL;
    }

    return out;
}

void free(float *ptr)
{
    free(ptr);
}