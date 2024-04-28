#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// #define ROWS_A 1024
// #define COLS_A 2048
// #define COLS_B 1024

#define A_ROWS 20
#define A_COLS 10
#define B_ROWS 10
#define B_COLS 20

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

    // generate random float arrays
    generate_random_float_array(A, A_ROWS * A_COLS);
    generate_random_float_array(B, A_COLS * B_COLS);

    // for (int tmp1=0; tmp1<A_COLS; tmp1++) {
    //     printf("%f ", A[tmp1]);
    // }
    // printf("\n");
    // for (int tmp2=0; tmp2<B_ROWS; tmp2++) {
    //     printf("%f ", B[tmp2*B_COLS]);
    // }
    // printf("\n");

    for (int i=0; i<A_ROWS; i++) {
        for (int j=0; j<B_COLS; j++) {
            C[i*B_COLS + j] = 0.0;
            for (int k=0; k<A_COLS; k++) {
                C[i*B_COLS + j] += A[i*A_COLS + k] * B[k*B_COLS + j];
            }
        }
    }

    clock_t start_time1 = clock();
    // perform matrix multiplication
    for (int i = 0; i < A_ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            C[i * B_COLS + j] = 0.0f;
            for (int k = 0; k < A_COLS; k++) {
                C[i * B_COLS + j] += A[i * A_COLS + k] * B[k * B_COLS + j];
            }
        }
    }
    clock_t end_time1 = clock();
    double elapsed_time1 = (double)(end_time1 - start_time1) / CLOCKS_PER_SEC;
    printf("navie matmul took %.6f seconds.\n", elapsed_time1);
    
    // printf("%f", C[0]);
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