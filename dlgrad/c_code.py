
class C:
    # TODO: Pass all args when compling itself ?
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

    def _ones_buffer() -> str:
        prg = """
        #include <stdio.h>
        #include <stdlib.h>

        float *create_ones_buffer(int length) 
        {
            float *c = malloc(length * sizeof(float));
            if (c == NULL) 
                return NULL;
            
            for (int i=0; i<length; i++)
                c[i] = 1.0f;
            
            return c;
        }
        """
        return prg 
    
    def _add_axis1(dtype: str, out_len: int) -> str:
        prg = f"""
        #include <stdio.h>
        #include <stdlib.h> 

        {dtype} *add_with_broadcasting(float* x, float* y, int len_a, int len_b) 
        {{
            int b_idx = 0;
            {dtype} *out = malloc({out_len} * sizeof({dtype}));
            if (out == NULL) 
                return NULL;

            for (int ptr_a = 0; ptr_a < len_a; ++ptr_a) {{
                b_idx = ptr_a % len_b;

                out[ptr_a] = x[ptr_a] + y[b_idx];
            }}

            return out;
        }}
        """
        return prg
    
    def _add_axis0(dtype: str, out_len: int) -> str:
        prg = f"""
        #include <stdio.h>
        #include <stdlib.h> 

        {dtype} *add_with_broadcasting(float* x, float* y, int len_a, int len_b, int ncol) 
        {{
            int b_idx = 0;
            {dtype} *out = malloc({out_len} * sizeof({dtype}));
            if (out == NULL) 
                return NULL;

            for (int ptr_a = 0; ptr_a < len_a; ++ptr_a) {{
                if (i % ncol == 0 && i != 0)
                    b_idx++;

                out[ptr_a] = x[ptr_a] + y[b_idx];
            }}

            return out;
        }}
        """
        return prg

    def _sum_axis0(dtype: str) -> str:
        prg = f"""
        #include <stdio.h>
        #include <stdlib.h>

        {dtype} *sum_axis0({dtype} *a, int len, int nrows, int ncols) {{
            {dtype} sum = 0.0f;
            {dtype} *res = malloc(ncols * sizeof({dtype}));
            
            for (int i=0; i<ncols; i++) {{
                sum = 0.0f;
                for (int j=i; j<len; j+=ncols) {{
                    sum += a[j];
                }}
                res[i] = sum;
            }}

            return res;
        }}
        """

        return prg

    def _sum_axis1(dtype: str) -> str:
        prg = f"""
        #include <stdio.h>
        #include <stdlib.h>

        {dtype} *sum_axis1({dtype} *a, int len, int nrows, int ncols) {{
            {dtype} sum = 0.0f;
            {dtype} *res = malloc(nrows * sizeof({dtype}));
            
            int j = 0;
            int idx = 0;
            for (int i=0; i<len; i+=ncols) {{
                sum = 0.0f;
                for (int j=i; j<(i+ncols); j++) {{
                    sum += a[j];
                }}
                res[idx++] = sum;
            }}

            return res;
        }}
        """

        return prg
    
    def _sum(dtype: str):
        prg = f"""
        #include <stdio.h>
        #include <stdlib.h>

        {dtype} sum({dtype} *a, int len) {{
            {dtype} sum = 0.0f;
            
            for (int i=0; i<len; i++) {{
                sum += a[i];
            }}

            return sum;
        }}
        """

        return prg

    def _matmul(dtype: str):
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
                int j = n%rows;
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