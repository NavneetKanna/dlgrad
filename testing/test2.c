#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// #define ROWS_A 1024
// #define COLS_A 2048
// #define COLS_B 1024

#define A_ROWS 1024
#define A_COLS 2048
#define B_ROWS 2048
#define B_COLS 1024

void generate_random_float_array(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (float)rand() / (float)RAND_MAX; // generate random float between 0 and 1
    }
}

int main() {
    // seed random number generator
    srand(time(NULL));

    float *A = (float *)malloc(A_ROWS * A_COLS * sizeof(float));
    float *B = (float *)malloc(B_ROWS * B_COLS * sizeof(float));
    float *C = (float *)malloc(A_ROWS * B_COLS * sizeof(float));

    generate_random_float_array(A, A_ROWS * A_COLS);
    generate_random_float_array(B, A_COLS * B_COLS);

    // for (int tmp=0; tmp<A_ROWS*B_COLS; tmp++) {
    //     C[tmp] = 0.0;
    // }

    clock_t start_time2 = clock();
    // loop interchange 
    for (int i = 0; i < A_ROWS; i++) {
        for (int k = 0; k < A_COLS; k++) {
            for (int j = 0; j < B_COLS; j++) {
                C[i * B_COLS + j] += A[i * A_COLS + k] * B[k * B_COLS + j];
            }
        }
    }
    clock_t end_time2 = clock();
    double elapsed_time2 = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
    printf("loop interchange matmul took %.6f seconds.\n", elapsed_time2);

    // printf("%f\n", C[0]);
    // print result (optional)
    // for (int i = 0; i < ROWS_A; i++) {
    //     for (int j = 0; j < COLS_B; j++) {
    //         printf("%f ", c[i * COLS_B + j]);
    //     }
    //     printf("\n");
    // }

    // free memory
    free(A);
    free(B);
    free(C);

    return 0;
}