from dlgrad.tensor import Tensor

"""
// Online C compiler to run C program online
#include <stdio.h>

void add(float a[], float b[], int k, int stride, int len_a, int len_b) {
    int ptr_a = 0;
    float res[len_a];
    int ptr_c = 0;
    int idx = -1;
    
    // k can be no of rows or cols depending on axis
    for (int i=0; i<k; i++) {
        idx = ptr_a;
        for (int ptr_b=0; ptr_b<len_b; ptr_b++) {
            printf("idx: %d ", idx);
            int sum = a[idx] + b[ptr_b];
            printf("sum: %d\n", sum);
            res[ptr_c] = sum;
            ptr_c += 1;
             idx += stride;
            
        }
        ptr_a += len_b;
        printf("ptr_a: %d \n", ptr_a);
    }
    
    for (int i=0; i<len_a; i++) {
        printf("%f ", res[i]);
    }
}

int main() {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    
    add(a, b, 3, 3, 9, 3);

    return 0;
}

"""