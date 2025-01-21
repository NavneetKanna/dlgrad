#ifndef ALLOCATE
#define ALLOCATE

float *uninitialized_memory(size_t nbytes);
float *initialized_memory(size_t num, size_t size);
float *init_with_scalar(size_t nbytes, int numel, int scalar);
void free_ptr(float *ptr);

#endif