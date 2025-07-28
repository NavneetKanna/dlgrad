// Online C compiler to run C program online
#include <stdio.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

void max_3d(float *x, float *out, int *stride, int *shape, int dim) {
    int out_ptr = -1;
    int last_start = 0;
    bool copy = true;
    
    for (int i=0; i<36; i++) {
        if (i % (stride[dim]*shape[dim]) == 0 && i != 0) {
            last_start += stride[dim];
            out_ptr = last_start;
            copy = true;
        } else if (i % stride[dim] == 0 && i != 0) {
            out_ptr -= stride[dim] - 1;
            copy = false;
        } else {
            out_ptr += 1;
        }
        
        // printf("i %d out_ptr %d \n", i, out_ptr);

        if (copy) {
            out[out_ptr] = x[i];
        } else {
            if (x[i] > out[out_ptr]) {
                out[out_ptr] = x[i];
            }
        }
    }
}
        

int main() {
    float x[36] = {
       0.1683, 0.9320, 0.8233, 0.0459, 0.3090, 0.9374, 0.9650, 0.4906, 0.4394,
        0.5902, 0.9465, 0.3153, 0.0948, 0.1133, 0.2267, 0.4538, 0.6649, 0.9079,
        0.1950, 0.0337, 0.7309, 0.9361, 0.7790, 0.4404, 0.1499, 0.8373, 0.8189,
        0.8005, 0.1392, 0.9983, 0.1907, 0.6233, 0.3478, 0.5163, 0.9265, 0.7512
    };
    
    int stride[4] = {12, 4, 2, 1};
    int shape[4] = {3, 3, 2, 2};
    
    float out[12];
    
    max_3d(x, out, stride, shape, 1);
    
    for (int i=0; i<12; i++) {
        printf("%f ", out[i]);
    }
    

    return 0;
}