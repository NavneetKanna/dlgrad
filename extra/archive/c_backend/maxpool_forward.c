/*
#include <stdio.h>


void maxpool_forward_c(float* array, float* res, int* max_index_array, int BS, int channels, int rows, int cols, int kernal_size, int stride) {
    int res_idx = 0;
    int max_idx = 0;
    for(int bs = 0; bs < BS; bs ++) {
        for(int channel = 0; channel < channels; channel++){
            for(int row = 0; row <= rows - kernal_size; row += stride){
                
                if(row > rows-kernal_size){
                    continue;
                }
                for(int col = 0; col <= cols - kernal_size; col += stride){
                    if(col > cols-kernal_size){
                        continue;
                    }
                    // printf("-------------\n");
                    int bs_offset = bs * channels * rows * cols;
                    int channel_offset = channel * rows * cols;
                    
                    int row_offset = (row+kernal_size) ;
                    int col_offset = (col+kernal_size);
                    float max = array[bs_offset + channel_offset + (row*cols) + col];
                    int temp = bs_offset + channel_offset + (row*cols) + col;
                    // printf("%d \n", temp);
                    for(int i = row; i < row_offset; i++){
                        for(int j = col; j < col_offset; j++){
                            int idx = bs_offset + channel_offset + (i*cols) + j;
                            float val = array[bs_offset + channel_offset + (i*cols) + j];
                            if(val > max){
                                max = val;
                                temp = bs_offset + channel_offset + (i*cols) + j;
                                // printf("%d \n", temp);
                            }
                        }
                    }
                    res[res_idx] = max;
                    max_index_array[res_idx] = temp;
                    res_idx += 1;
                }
                // printf("finished \n");
            }
        }
    }
}
*/