#ifndef ADD
#define ADD

float *add_2d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride, int yshape_len);
float *add_3d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride, int yshape_len);
void free_add(float *ptr);

#endif 