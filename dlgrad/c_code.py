
class C:
    @staticmethod
    def _create_empty_buffer(len):
        pass
    
    @staticmethod
    def _random_buffer():
        # TODO: is this float32 ?
        # TODO: is drand48 available on unix and win ?
        # TODO: check malloc
        prg = """
        #include <stdio.h>
        #include <stdlib.h>
        #include <time.h>

        float *create_rand_buffer(int length) {
            float *data = malloc(length * sizeof(float));
            srand48(time(NULL));
            for (int i=0; i<length; i++) {
            data[i] = drand48();
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