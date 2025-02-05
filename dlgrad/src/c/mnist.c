#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>

// https://web.archive.org/web/20160828233817/http://yann.lecun.com/exdb/mnist/index.html
float *mnist_images_loader(char *path, uint32_t magic_number)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("Failed to open file\n");
        return NULL;
    }

    uint32_t magic_num, num_images, num_rows, num_cols;

    fread(&magic_num, 1, 4, fp);

    magic_num = ntohl(magic_num);

    if (magic_num != magic_number) {
        printf("Invalid magic number\n");
        fclose(fp);
        return NULL;
    }

    fread(&num_images, 1, 4, fp);
    fread(&num_rows, 1, 4, fp);
    fread(&num_cols, 1, 4, fp);
    num_images = ntohl(num_images);
    num_rows = ntohl(num_rows);
    num_cols = ntohl(num_cols);

   int out_size = num_images * num_rows * num_cols;

    float *out = (float*)malloc(out_size * sizeof(float));
    if (!out) {
        printf("Failed to allocate memory\n");
        fclose(fp);
        return NULL;
    }

    unsigned char pixel;
    for (int i=0; i<out_size; i++) {
        fread(&pixel, 1, 1, fp);
        out[i] = (float)pixel / 255.0f;
    }

    fclose(fp);

    return out;
}

float *mnist_labels_loader(char *path, uint32_t magic_number)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("Failed to open file\n");
        return NULL;
    }

    uint32_t magic_num, num_images;

    fread(&magic_num, 1, 4, fp);

    magic_num = ntohl(magic_num);

    if (magic_num != magic_number) {
        printf("Invalid magic number\n");
        fclose(fp);
        return NULL;
    }

    fread(&num_images, 1, 4, fp);
    num_images = ntohl(num_images);

    int out_size = num_images;

    float *out = (float*)malloc(out_size * sizeof(float));
    if (!out) {
        printf("Failed to allocate memory\n");
        fclose(fp);
        return NULL;
    }

    unsigned char labels;
    for (int i=0; i<out_size; i++) {
        fread(&labels, 1, 1, fp);
        out[i] = (float)labels;
    }

    fclose(fp);

    return out;
}