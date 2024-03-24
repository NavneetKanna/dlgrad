#include <stdlib.h>
#include <stdio.h>

void free_buf(void *data) {
    printf("freeing data\n");
    free(data);
}