#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <complex>
#include "sparse.h"
#include "info_threads.h"

namespace py = pybind11;

PYBIND11_MODULE(mtspy_cpp, m)
{
    m.def("mat_vec_f", &mat_vec<float>, py::call_guard<py::gil_scoped_release>());
    m.def("mat_vec_d", &mat_vec<double>, py::call_guard<py::gil_scoped_release>());
    m.def("mat_vec_cf", &mat_vec<std::complex<float>>, py::call_guard<py::gil_scoped_release>());
    m.def("mat_vec_cd", &mat_vec<std::complex<double>>, py::call_guard<py::gil_scoped_release>());

    m.def("max_threads", &max_threads);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}