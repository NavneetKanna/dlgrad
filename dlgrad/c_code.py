
class C:
    @staticmethod
    def _create_empty_buffer(len):
        pass
    
    @staticmethod
    def _random_buffer() -> str:
        # TODO: is this float32 ?
        # TODO: is srand48 available on unix and win ?
        # TODO: is this uniform distribution ?
        prg = """
        #include <stdio.h>
        #include <stdlib.h>
        #include <time.h>

        float *create_rand_buffer(int length) {
            float *data = malloc(length * sizeof(float));
            if (data == NULL)
                return NULL;

            srand48(time(NULL));
            for (int i=0; i<length; i++) {
                data[i] = (((float)rand() / (float)RAND_MAX));
            }
            return data;
        }
        """ 
        return prg

    def _add(dtype: str, out_len: int) -> str:
        # TODO: Wrong here, arr should also be dtype
        prg = f"""
        #include <stdio.h>
        #include <stdlib.h> 

        {dtype} *add(float *x, float *y) 
        {{
            {dtype} *out = malloc({out_len} * sizeof({dtype}));
            if (out == NULL) 
                return NULL;
            for (int i=0; i<{out_len}; i++) {{
                out[i] = x[i] + y[i];
            }}
            return out;
        }}
        """
        return prg
    
    def _matmul(dtype: str):
        # TODO: Change to x and y maybe:
        prg = f"""
        #include <stdio.h>
        #include <stdlib.h>
        
        // loop interchange matmul
        {dtype} *matmul({dtype} *a, {dtype} *b, int A_ROWS, int A_COLS, int B_COLS) 
        {{
            {dtype} *c = ({dtype} *)malloc(A_ROWS * B_COLS * sizeof({dtype}));
            for (int i=0; i<A_ROWS; i++) {{
                for (int k=0; k<A_COLS; k++) {{
                   for (int j=0; j<B_COLS; j++) {{
                        c[i*B_COLS + j] += a[i*A_COLS + k] * b[k*B_COLS + j];
                    }}
                }}
            }}
            return c;
        }}
        """
        return prg

    def _free() -> str:
        prg = """
        #include <stdlib.h>
        #include <stdio.h>

        void free_buf(void *data) 
        {
            free(data);
        }
        """
        return prg