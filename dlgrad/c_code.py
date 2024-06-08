
class C:
    @staticmethod
    def _random_buffer() -> str:
        prg = """
        #include <stdlib.h>
        #include <time.h>
        #include <inttypes.h>
        #include <stdio.h>
        #include <stdint.h>

        /*
         * This block of code generates random numbers
         * from a uniform distribution on the interval [low, high).
         * It uses xoroshiro64* (https://prng.di.unimi.it/xoroshiro64star.c).
         */
        static inline uint32_t rotl(const uint32_t x, int k) {
            return (x << k) | (x >> (32 - k));
        }

        static uint32_t s[2];

        uint32_t next(void) {
            const uint32_t s0 = s[0];
            uint32_t s1 = s[1];
            const uint32_t result = s0 * 0x9E3779BB;

            s1 ^= s0;
            s[0] = rotl(s0, 26) ^ s1 ^ (s1 << 9); // a, b
            s[1] = rotl(s1, 13); // c

            return result;
        }

        float random_uniform(float a, float b) {
            uint32_t rnd = next();
            float normalized = rnd / (float)UINT32_MAX; // Normalize to [0, 1)
            float new = (b-a)*(normalized) + a; // transform to range (a, b)
            return new; 
        }

        float *create_rand_buffer(int length, float low, float high) {
            s[0] = (uint32_t)time(NULL);
            s[1] = (uint32_t)time(NULL) + 1; 

            float *data = malloc(length * sizeof(float));
            if (data == NULL)
                return NULL;

            for (int i=0; i<length; i++) {
                float random_value = random_uniform(low, high);
                data[i] = random_value;
            }

            return data;
        }
        """ 
        return prg

    def _add(dtype: str, out_len: int) -> str:
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
            if (c == NULL) 
                return NULL;

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

    def _transpose(dtype):
        prg = f"""
        #include <stdio.h>
        #include <stdlib.h>

        // https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
        float *transpose(float *src, int rows, int cols) 
        {{
            float *dst = ({dtype} *)malloc((rows*cols) * sizeof({dtype}));
            if (dst == NULL) 
                return NULL;

            for(int n = 0; n<rows*cols; n++) {{
                int i = n/rows;
                int j = n%cols;
                dst[n] = src[cols*j + i];
            }}
            return dst;
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