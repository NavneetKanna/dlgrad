#ifndef ALLOCATE
#define ALLOCATE

float *uninitialized_memory(size_t num);
float *initialized_memory(size_t num);
void free_ptr(float *ptr);

#endif