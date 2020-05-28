#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <complex>
#include "sparse.h"
#include "thread_control.h"

namespace py = pybind11;

PYBIND11_MODULE(mtspy_cpp, m)
{
    m.def("mat_vec_f", &mat_vec<float>, py::call_guard<py::gil_scoped_release>());
    m.def("mat_vec_d", &mat_vec<double>, py::call_guard<py::gil_scoped_release>());
    m.def("mat_vec_cf", &mat_vec<std::complex<float>>, py::call_guard<py::gil_scoped_release>());
    m.def("mat_vec_cd", &mat_vec<std::complex<double>>, py::call_guard<py::gil_scoped_release>());

    // Thread Control:
    m.def("get_max_threads", &get_max_threads);
    m.def("get_num_threads", &get_num_threads);
    m.def("set_num_threads", &set_num_threads);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}