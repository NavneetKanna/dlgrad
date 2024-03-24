#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float *add(float *arr1, float *arr2, int len) {
    float *result = malloc(len * sizeof(int));
    for (int i=0; i<len; i++) {
        result[i] = arr1[i] + arr2[i];
    }

    return result;
}

void free_buf(void *data) {
    printf("freeing data\n");
    free(data);
}