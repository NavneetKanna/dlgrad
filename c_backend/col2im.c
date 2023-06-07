#include <stdio.h>

/*

(1, 2, 4, 4)                               (2, 2, 4, 4)

    | im2col                                    | im2col

 (9, 8)                                        (18, 8)



(1, 2, 4, 4)                                (2, 2, 4, 4)
 
    | col2cim                                   | col2im

  (9, 8)                                       (18, 8)


*/


void col2im_c(float* array, float* res, int BS, int channels, int rows, int cols, int kernal_size, int stride) {
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
                            // printf("idx %d ", batch_offset + channel_offset + (i*cols) + j);
                            // printf("res %f ", res[res_idx]);
                            array[batch_offset + channel_offset + (i*cols) + j] += res[res_idx];
                            // printf("%f \n", array[batch_offset + channel_offset + (i*cols) + j]);
                            // printf("val %f\n", array[batch_offset + channel_offset + (i*cols) + j]);
                            // float  val = array[batch_offset + channel_offset + (i*cols) + j];
                            // res[res_idx] = val;
                            res_idx += 1;
                        }
                    }
                }
            }
        }   
    }






}