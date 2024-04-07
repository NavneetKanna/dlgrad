
class C:
    @staticmethod
    def _create_empty_buffer(len):
        pass
    
    @staticmethod
    def _random_buffer():
        # TODO: is this float32 ?
        # TODO: is srand48 available on unix and win ?
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

    def _add(dtype, out_len):
        prg = f"""
        #include <stdio.h>
        #include <stdlib.h> 

        {dtype} *add(float *x, float *y) {{
            {dtype} *out = malloc({out_len} * sizeof({dtype}));
            for (int i=0; i<{out_len}; i++) {{
                out[i] = x[i] + y[i];
            }}
            return out;
        }}
        """
        return prg

    def _free():
        prg = """
        #include <stdlib.h>
        #include <stdio.h>

        void free_buf(void *data) {
            free(data);
        }
        """
        return prg