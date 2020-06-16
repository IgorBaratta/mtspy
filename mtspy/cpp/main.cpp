#include "eigen.hpp"
#include "sparse.hpp"
#include "thread_control.hpp"
#include <complex>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mtspy_cpp, m)
{
    //-------------------------------------------------------------------------------------//
    // Sparse Matrix Vector Product (32 bit indices)
    m.def("spmv", &SpMV<float, std::int32_t>, py::return_value_policy::move);
    m.def("spmv", &SpMV<double, std::int32_t>, py::return_value_policy::move);
    m.def("spmv", &SpMV<std::complex<float>, std::int32_t>, py::return_value_policy::move);
    m.def("spmv", &SpMV<std::complex<double>, std::int32_t>, py::return_value_policy::move);
    // Sparse Matrix Vector Product (64 bit indices)
    m.def("spmv", &SpMV<float, std::int64_t>, py::return_value_policy::move);
    m.def("spmv", &SpMV<double, std::int64_t>, py::return_value_policy::move);
    m.def("spmv", &SpMV<std::complex<float>, std::int64_t>, py::return_value_policy::move);
    m.def("spmv", &SpMV<std::complex<double>, std::int64_t>, py::return_value_policy::move);

    //-------------------------------------------------------------------------------------//
    // Sparse Matrix - Dense Matrix Product (32 bit indices)
    m.def("spmm", &SpMM<float, std::int32_t>, py::return_value_policy::move);
    m.def("spmm", &SpMM<double, std::int32_t>, py::return_value_policy::move);
    m.def("spmm", &SpMM<std::complex<float>, std::int32_t>, py::return_value_policy::move);
    m.def("spmm", &SpMM<std::complex<double>, std::int32_t>, py::return_value_policy::move);
    // Sparse Matrix - Dense Matrix Product (64 bit indices)
    m.def("spmm", &SpMM<float, std::int64_t>, py::return_value_policy::move);
    m.def("spmm", &SpMM<double, std::int64_t>, py::return_value_policy::move);
    m.def("spmm", &SpMM<std::complex<float>, std::int64_t>, py::return_value_policy::move);
    m.def("spmm", &SpMM<std::complex<double>, std::int64_t>, py::return_value_policy::move);

#ifdef USE_EIGEN_BACKEND
    //-------------------------------------------------------------------------------------//
    // Sparse Matrix - Dense Matrix Product  using eigen backend (32 bit indices)
    m.def("spmm_eigen", &SpMM_eigen<float, std::int32_t>, py::return_value_policy::move);
    m.def("spmm_eigen", &SpMM_eigen<double, std::int32_t>, py::return_value_policy::move);
    m.def("spmm_eigen", &SpMM_eigen<std::complex<float>, std::int32_t>, py::return_value_policy::move);
    m.def("spmm_eigen", &SpMM_eigen<std::complex<double>, std::int32_t>, py::return_value_policy::move);

    // Sparse Matrix - Dense Matrix Product  using eigen backend (64 bit indices)
    m.def("spmm_eigen", &SpMM_eigen<float, std::int64_t>, py::return_value_policy::move);
    m.def("spmm_eigen", &SpMM_eigen<double, std::int64_t>, py::return_value_policy::move);
    m.def("spmm_eigen", &SpMM_eigen<std::complex<float>, std::int64_t>, py::return_value_policy::move);
    m.def("spmm_eigen", &SpMM_eigen<std::complex<double>, std::int64_t>, py::return_value_policy::move);
#endif

    m.def("has_eigen", &has_eigen);

    //-------------------------------------------------------------------------------------//
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