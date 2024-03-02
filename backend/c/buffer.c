#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float *create_rand_buffer(int len) {
    float *data = malloc(len * sizeof(float));

    return data;
}

void free_buf(void *data) {
    printf("freeing data\n");
    free(data);
}

/*
clang -shared -o test.so buffer.c 
*/