#include <pybind11/pybind11.h>
#include "add.h"
#include "sub.h"

namespace py = pybind11;

PYBIND11_MODULE(interface, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
    m.def("sub", &sub, "A function that adds two numbers");
}
