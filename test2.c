#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BS 4
#define C 3
#define W 4096
#define H 4096

#define SIZE ((size_t)BS * C * W * H)

// Function to generate random floats between 0 and 1
void fill_random(float *arr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        arr[i] = (float)rand() / RAND_MAX;
    }
}

// High resolution time difference in milliseconds
double time_diff_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1e6;
}

int main() {
    float *arr = malloc(sizeof(float) * SIZE);
    float *arr2 = malloc(sizeof(float) * SIZE);
    if (!arr || !arr2) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    srand(time(NULL));
    fill_random(arr, SIZE);
    fill_random(arr2, SIZE);

    float sum = 0.0f;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < SIZE; i++) {
        sum += (arr[i] + arr2[i]);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.2f ms\n", time_diff_ms(start, end));

    printf("Sum: %.4f\n", sum); // Optional: just to ensure result is used
    free(arr);
    free(arr2);
    return 0;
}

// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>

// #define BS 4
// #define C 3
// #define W 4096
// #define H 4096

// #define SIZE ((size_t)BS * C * W * H)

// // Fixed strides
// #define STRIDE_BS 50331648  // C * W * H
// #define STRIDE_C  16777216  // W * H
// #define STRIDE_W  4096      // H

// // Blocking sizes
// #define BW 32
// #define BH 32

// // Generate random floats
// void fill_random(float *arr, size_t size) {
//     for (size_t i = 0; i < size; i++) {
//         arr[i] = (float)rand() / RAND_MAX;
//     }
// }

// // Time diff in ms
// double time_diff_ms(struct timespec start, struct timespec end) {
//     return (end.tv_sec - start.tv_sec) * 1000.0 +
//            (end.tv_nsec - start.tv_nsec) / 1e6;
// }

// int main() {
//     float *arr = malloc(sizeof(float) * SIZE);
//     float *arr2 = malloc(sizeof(float) * SIZE);
//     if (!arr || !arr2) {
//         fprintf(stderr, "Memory allocation failed\n");
//         return 1;
//     }

//     srand(time(NULL));
//     fill_random(arr, SIZE);
//     fill_random(arr2, SIZE);

//     float sum = 0.0f;
//     struct timespec start, end;
//     clock_gettime(CLOCK_MONOTONIC, &start);

//     for (int bs = 0; bs < BS; bs++) {
//         for (int c = 0; c < C; c++) {
//             for (int wb = 0; wb < W; wb += BW) {
//                 for (int hb = 0; hb < H; hb += BH) {
//                     for (int w = wb; w < wb + BW; w++) {
//                         for (int h = hb; h < hb + BH; h++) {
//                             size_t offset = ((size_t)bs * STRIDE_BS) +
//                                             ((size_t)c * STRIDE_C) +
//                                             ((size_t)w * STRIDE_W) +
//                                             h;
//                             sum += arr[offset] + arr2[offset];
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     clock_gettime(CLOCK_MONOTONIC, &end);
//     printf("Time taken: %.2f ms\n", time_diff_ms(start, end));
//     printf("Sum: %.4f\n", sum); // Optional: just to ensure result is used

//     free(arr);
//     free(arr2);
//     return 0;
// }
