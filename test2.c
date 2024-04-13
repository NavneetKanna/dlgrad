#include <Accelerate/Accelerate.h>

int main() {
    float a[2] = {1.0, 2.0};
    float b[2] = {1.0, 2.0};
    float out[2];
    vDSP_vadd(a, 1, b, 1, out, 1, 2);

    return 0;
}