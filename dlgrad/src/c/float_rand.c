#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

#include "pcg_basic.h"
#include "float_rand.h"

pcg32_random_t rng;

float *uniform(int numel, float low, float high) {
    int fd = open("/dev/random", O_RDONLY);
    if (fd < 0) {
        return NULL;
    }

    uint64_t seed;
    ssize_t bytes_read = read(fd, &seed, sizeof(seed));
    if (bytes_read != sizeof(seed)) {
        return NULL;
    }

    close(fd);

    pcg32_srandom_r(&rng, seed, (intptr_t)&rng);

    float *out = malloc(numel * sizeof(float));

    for (int i= 0; i < numel; i++) {
        double d = ldexp(pcg32_random_r(&rng), -32);
        float f = (float) d;

        if (low == 0.0f && high == 1.0f) {
            out[i] = f;
        } else {
            out[i] = low + (high - low) * f;
        }
    }
    return out;
}

void free_uniform(float* ptr) {
    free(ptr);
}
