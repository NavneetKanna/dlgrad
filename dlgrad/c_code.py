
class C:
    @staticmethod
    def _create_empty_buffer(len):
        prg = ...

    @staticmethod
    def _random_buffer():
        # TODO: is this float32 ?
        prg = """
        #include <stdio.h>
        #include <stdlib.h>
        #include <time.h>

        float *create_rand_buffer(int length) {
            float *data = malloc(length * sizeof(float));
            srand(time(NULL));
            for (int i=0; i<length; i++) {
                data[i] = (float)rand() / (float)RAND_MAX;
            }
            return data;
        }
        """ 
        return prg

    def _free():
        prg = """
        #include <stdlib.h>
        #include <stdio.h>

        void free_buf(void *data) {
            printf("freeing data");
            free(data);
        }
        """
        return prg