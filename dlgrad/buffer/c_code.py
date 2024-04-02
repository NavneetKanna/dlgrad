
class C:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _create_empty_buffer(len):
        prg = """
        #include <stdio.h>
        
        float *

        """ 


    @staticmethod
    def _random_buffer(len):
        prg = """
        #include <stdio.h>
        #include <stdlib.h>
        #include <time.h>

        float *create_rand_buffer(int {len}) {
            float *data = malloc(len * sizeof(float));
            srand(time(NULL));
            for (int i=0; i<len; i++) {
                data[i] = (float)rand() / (float)RAND_MAX;
            }
            return data;
        }
        """ 
        return prg