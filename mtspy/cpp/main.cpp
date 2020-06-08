#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <complex>
#include "sparse.h"
#include "thread_control.h"

namespace py = pybind11;

PYBIND11_MODULE(mtspy_cpp, m)
{
    // spmv using 32 bit intergers for indices:
    m.def("spmv", &SpMV<float, std::int32_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmv", &SpMV<double, std::int32_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmv", &SpMV<std::complex<float>, std::int32_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmv", &SpMV<std::complex<double>, std::int32_t>, py::call_guard<py::gil_scoped_release>());

    // spmv using 64 bit intergers for indices:
    m.def("spmv", &SpMV<float, std::int64_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmv", &SpMV<double, std::int64_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmv", &SpMV<std::complex<float>, std::int64_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmv", &SpMV<std::complex<double>, std::int64_t>, py::call_guard<py::gil_scoped_release>());

    // SpMM using 32 bit intergers for indices:
    m.def("spmm", &SpMM<float, std::int32_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmm", &SpMM<double, std::int32_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmm", &SpMM<std::complex<float>, std::int32_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmm", &SpMM<std::complex<double>, std::int32_t>, py::call_guard<py::gil_scoped_release>());

    // SpMM using 32 bit intergers for indices:
    m.def("spmm", &SpMM<float, std::int64_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmm", &SpMM<double, std::int64_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmm", &SpMM<std::complex<float>, std::int64_t>, py::call_guard<py::gil_scoped_release>());
    m.def("spmm", &SpMM<std::complex<double>, std::int64_t>, py::call_guard<py::gil_scoped_release>());

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