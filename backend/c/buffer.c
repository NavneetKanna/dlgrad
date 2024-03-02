#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void create_rand_buffer(int bs, int ch, int w, int h) {
    float random_float;
    int i, j, k, l;

    srand(time(NULL));

    int count = !(bs == 0) + !(ch == 0) + !(w == 0) + !(h == 0);

    switch (count) {
        case 4: {
            float data[bs][ch][w][h];
            for (i = 0; i < bs; i++) {
                for (j = 0; j < ch; j++) {
                    for (k = 0; k < w; k++) {
                        for (l = 0; l < h; l++) {
                            data[i][j][k][l] = (float)rand() / (float)RAND_MAX;
                        }
                    }
                }
            }
            break;
        }
        case 3: {
            float data[ch][w][h];
            for (i = 0; i < ch; i++) {
                for (j = 0; j < w; j++) {
                    for (k = 0; k < h; k++) {
                        data[i][j][k] = (float)rand() / (float)RAND_MAX;
                    }
                }
            }
            break;
        }
        case 2: {
            float data[w][h];
            for (i = 0; i < w; i++) {
                for (j = 0; j < h; j++) {
                    printf("in for loop ");
                    data[i][j] = (float)rand() / (float)RAND_MAX;
                }
            }
            break;
        }
        case 1: {
            float data[h];
            for (i = 0; i < h; i++) {
                data[i] = (float)rand() / (float)RAND_MAX;
            }
            break;
        }
    }
}


/*
clang -shared -o test.so buffer.c 
*/