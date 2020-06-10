#ifdef USE_EIGEN_BACKEND

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <omp.h>

template <typename T, typename I>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
SPMV_eigen(I rows, I cols, I nnz,
           const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> &data,
           const Eigen::Ref<Eigen::Matrix<I, Eigen::Dynamic, 1>> &indptr,
           const Eigen::Ref<Eigen::Matrix<I, Eigen::Dynamic, 1>> &indices,
           const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> dense)
{
    // Currently only work with row-major sparse matrix in parallel
    Eigen::Map<const Eigen::SparseMatrix<T, Eigen::RowMajor>> sm1(rows, cols, nnz, indptr.data(), indices.data(), data.data());
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> output = sm1 * dense;
    pybind11::gil_scoped_acquire acquire;
    return output;
}

#endif