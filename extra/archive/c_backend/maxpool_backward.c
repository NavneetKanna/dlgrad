/*
#include<stdio.h>


void maxpool_backward_c(float* array, float* data_grad, float* out_grad, int* max_index_array, int BS, int channels, int rows, int cols, int kernal_size, int stride){
    int temp = 0;
    // printf("%d ", channels);
    for(int bs = 0; bs < BS; bs ++) {
        // printf("bs %d\n", bs);
        for(int channel = 0; channel < channels; channel++){
            // printf("channel %d\n", channel);
            for(int row = 0; row <= rows - kernal_size; row += stride){
                // printf("row %d\n", row);
                if(row > rows-kernal_size){
                    continue;
                }
                for(int col = 0; col <= cols - kernal_size; col += stride){
                    // printf("col %d\n", col);
                    if(col > cols-kernal_size){
                        continue;
                    }
                    // printf("-----------\n");
                    int bs_offset = bs * channels * rows * cols;
                    int channel_offset = channel * rows * cols;
                    
                    int row_offset = (row+kernal_size) ;
                    int col_offset = (col+kernal_size);
                    // float max = array[bs_offset + channel_offset + (row*cols) + col];
                    for(int i = row; i < row_offset; i++){
                        for(int j = col; j < col_offset; j++){
                            int u = bs_offset + channel_offset + (i*cols) + j;
                            // printf("%d %d\n", max_index_array[temp], u);
                            if(max_index_array[temp] == u) data_grad[u] += out_grad[temp];
                            // else data_grad[u] = 0;
                        }
                    }
                    temp += 1; 
                }
                // printf("finished \n");
            }
        }
    }


}
*/