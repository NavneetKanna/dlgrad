#ifndef ADD
#define ADD

float *add(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride, int ndim);
float *add_2d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride);
int get_y_idx_2d(int dim1, int dim2, int *yshape, int *ystride);
float *add_3d(float *x, float *y, int numel, int *xshape, int *yshape, int *xstride, int *ystride);
int get_y_idx_3d(int dim1, int dim2, int dim3, int *yshape, int *ystride);

#endif 