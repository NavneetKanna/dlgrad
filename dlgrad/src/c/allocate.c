#include <stdlib.h>
#include "allocate.h"


// dlgrad only supports float, hence it is ok to have the return type as float
float *uninitialized_memory(size_t nbytes)
{
    float *out = malloc(nbytes);
    if (out == NULL) {
        return NULL;
    }

    return out;
}

float *initialized_memory(size_t num, size_t size)
{
    float *out = calloc(num, size);
    if (out == NULL) {
        return NULL;
    }

    return out;
}

float *init_with_scalar(size_t nbytes, int numel, int scalar)
{
    float *out = malloc(nbytes);
    if (out == NULL) {
        return NULL;
    }

    for (int i=0; i<numel; i++) {
        out[i] = scalar;
    }

    return out;
}

void free_ptr(float *ptr)
{
    free(ptr);
}