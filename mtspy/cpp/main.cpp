#include "eigen.hpp"
#include "sparse.hpp"
#include "thread_control.hpp"
#include <complex>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mtspy_cpp, m)
{
    //====================================================================================//
    // Sparse Matrix Vector Product (32 bit indices)
    m.def("sparse_vec", &sparse_vec<float, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_vec", &sparse_vec<double, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_vec", &sparse_vec<std::complex<float>, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_vec", &sparse_vec<std::complex<double>, std::int32_t>, py::return_value_policy::move);
    // Sparse Matrix Vector Product (64 bit indices)
    m.def("sparse_vec", &sparse_vec<float, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_vec", &sparse_vec<double, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_vec", &sparse_vec<std::complex<float>, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_vec", &sparse_vec<std::complex<double>, std::int64_t>, py::return_value_policy::move);

    //====================================================================================//
    // Sparse Matrix - Dense Matrix Product (32 bit indices)
    m.def("sparse_dense", &sparse_dense<float, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_dense", &sparse_dense<double, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_dense", &sparse_dense<std::complex<float>, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_dense", &sparse_dense<std::complex<double>, std::int32_t>, py::return_value_policy::move);
    // Sparse Matrix - Dense Matrix Product (64 bit indices)
    m.def("sparse_dense", &sparse_dense<float, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_dense", &sparse_dense<double, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_dense", &sparse_dense<std::complex<float>, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_dense", &sparse_dense<std::complex<double>, std::int64_t>, py::return_value_policy::move);

    //====================================================================================//
    // Sparse Matrix - Sparse Matrix Product (32 bit indices)
    m.def("sparse_sparse", &sparse_sparse<float, std::int32_t>);
    m.def("sparse_sparse", &sparse_sparse<double, std::int32_t>);
    m.def("sparse_sparse", &sparse_sparse<std::complex<float>, std::int32_t>);
    m.def("sparse_sparse", &sparse_sparse<std::complex<double>, std::int32_t>);
    // Sparse Matrix - Sparse Matrix Product (64 bit indices)
    m.def("sparse_sparse", &sparse_sparse<float, std::int64_t>);
    m.def("sparse_sparse", &sparse_sparse<double, std::int64_t>);
    m.def("sparse_sparse", &sparse_sparse<std::complex<float>, std::int64_t>);
    m.def("sparse_sparse", &sparse_sparse<std::complex<double>, std::int64_t>);

#ifdef USE_EIGEN_BACKEND
    //====================================================================================//
    // Sparse Matrix - Dense Matrix Product  using eigen backend (32 bit indices)
    m.def("sparse_dense_eigen", &sparse_dense_eigen<float, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_dense_eigen", &sparse_dense_eigen<double, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_dense_eigen", &sparse_dense_eigen<std::complex<float>, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_dense_eigen", &sparse_dense_eigen<std::complex<double>, std::int32_t>, py::return_value_policy::move);

    // Sparse Matrix - Dense Matrix Product  using eigen backend (64 bit indices)
    m.def("sparse_dense_eigen", &sparse_dense_eigen<float, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_dense_eigen", &sparse_dense_eigen<double, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_dense_eigen", &sparse_dense_eigen<std::complex<float>, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_dense_eigen", &sparse_dense_eigen<std::complex<double>, std::int64_t>, py::return_value_policy::move);

    //====================================================================================//
    // Sparse Matrix - Sparse Matrix Product  using eigen backend (32 bit indices)
    m.def("sparse_sparse_eigen", &sparse_sparse_eigen<float, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_sparse_eigen", &sparse_sparse_eigen<double, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_sparse_eigen", &sparse_sparse_eigen<std::complex<float>, std::int32_t>, py::return_value_policy::move);
    m.def("sparse_sparse_eigen", &sparse_sparse_eigen<std::complex<double>, std::int32_t>, py::return_value_policy::move);

    // Sparse Matrix - Sparse Matrix Product  using eigen backend (64 bit indices)
    m.def("sparse_sparse_eigen", &sparse_sparse_eigen<float, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_sparse_eigen", &sparse_sparse_eigen<double, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_sparse_eigen", &sparse_sparse_eigen<std::complex<float>, std::int64_t>, py::return_value_policy::move);
    m.def("sparse_sparse_eigen", &sparse_sparse_eigen<std::complex<double>, std::int64_t>, py::return_value_policy::move);

#endif

    m.def("has_eigen", &has_eigen);

    //====================================================================================//
    // Thread Control:
    m.def("get_max_threads", &mtspy::threads::get_max_threads);
    m.def("get_num_threads", &mtspy::threads::get_num_threads);
    m.def("set_num_threads", &mtspy::threads::set_num_threads);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}