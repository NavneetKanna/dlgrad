#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float *create_rand_buffer(int len) {
    float *data = malloc(len * sizeof(float));
    srand(time(NULL));
    for (int i=0; i<len; i++) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
    return data;
}

void free_buf(void *data) {
    printf("freeing data\n");
    free(data);
}

/*
clang -shared -o test.so buffer.c 
*/