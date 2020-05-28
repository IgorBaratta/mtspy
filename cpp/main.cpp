#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "sparse.h"

namespace py = pybind11;

PYBIND11_MODULE(mtspy_cpp, m)
{
    m.def("mat_vec", &mat_vec, py::call_guard<py::gil_scoped_release>());

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}