#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "pcg_basic.h"
#include "float_rand.h"

pcg32_random_t rng;

float *uniform(int numel) {
    float *out = malloc(numel * sizeof(float));

    pcg32_srandom_r(&rng, time(NULL), (intptr_t)&rng);

    for (int i= 0; i < numel; i++) {
        double d = ldexp(pcg32_random_r(&rng), -32);
        float f = (float) d;
        out[i] = f;
    }

    return out;
}

void free_uniform(float* ptr) {
    free(ptr);
}
