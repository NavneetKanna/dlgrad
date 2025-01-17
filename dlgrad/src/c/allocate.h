#ifndef ALLOCATE
#define ALLOCATE

float *uninitialized_memory(size_t nbytes);
float *initialized_memory(size_t num, size_t size);
void free_ptr(float *ptr);

#endif