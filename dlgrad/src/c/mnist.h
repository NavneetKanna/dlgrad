#ifndef MNIST
#define MNIST

float *mnist_images_loader(char *path, uint32_t magic_number);
float *mnist_labels_loader(char *path, uint32_t magic_number);

#endif