#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <omp.h>

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
mat_vec(int rows, int cols, int nnz,
        Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> &data,
        Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, 1>> &indptr,
        Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, 1>> &indices,
        const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> v1)
{
    // Currently only work with row-major sparse matrix in parallel
    Eigen::Map<const Eigen::SparseMatrix<T, Eigen::RowMajor>> sm1(rows, cols, nnz, indptr.data(), indices.data(), data.data());
    Eigen::Matrix<T, Eigen::Dynamic, 1> v2 = sm1 * v1;
    pybind11::gil_scoped_acquire acquire;
    return v2;
}
