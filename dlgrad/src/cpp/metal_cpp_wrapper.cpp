#include "Metal.hpp"

extern "C" {
    void init()
    {
        MTL::CreateSystemDefaultDevice();
    }
}