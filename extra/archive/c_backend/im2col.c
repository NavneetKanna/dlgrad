/*
#include <stdio.h>


void im2col_c(float* array, float* res, int BS, int channels, int rows, int cols, int kernal_size, int stride) {
    int res_idx = 0;
    for (int bs = 0; bs < BS; bs++) {
        for (int row = 0; row <= rows - kernal_size; row += stride) {
            for (int col = 0; col <= cols - kernal_size; col += stride) {
                int batch_offset = bs * channels * rows * cols;
                int row_offset = (row+kernal_size);
                int col_offset = col+kernal_size;
                for(int c = 0; c < channels; c++){
                    int channel_offset = c * rows * cols;
                    for(int i = row; i < row_offset; i++){
                        for(int j = col; j < col_offset; j++){
                            float  val = array[batch_offset + channel_offset + (i*cols) + j];
                            res[res_idx] = val;
                            res_idx += 1;
                        }
                    }
                }
            }
        }   
    }
}
*/