#include <stdlib.h>
#include <stdio.h>

void create_rand_buffer(int bs, int ch, int w, int h) {
    printf("bs %d, ch %d, w %d, h %d\n", bs, ch, w, h);

}


/*
clang -shared -o test.so buffer.c 
*/