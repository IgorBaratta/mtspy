#ifdef USE_EIGEN_BACKEND

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <omp.h>
#include <pybind11/pybind11.h>

template <class ScalarType>
using dense_matrix = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename ScalarType, typename IndType>
dense_matrix<ScalarType>
SpMM_eigen(IndType rows, IndType cols, IndType nnz,
           const Eigen::Ref<Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>> &data,
           const Eigen::Ref<Eigen::Matrix<IndType, Eigen::Dynamic, 1>> &indptr,
           const Eigen::Ref<Eigen::Matrix<IndType, Eigen::Dynamic, 1>> &indices,
           const Eigen::Ref<dense_matrix<ScalarType>> dense)
{

    pybind11::gil_scoped_release release;
    // Eigen only work with row-major sparse matrix in parallel
    Eigen::Map<const Eigen::SparseMatrix<ScalarType, Eigen::RowMajor>> sm1(rows, cols, nnz, indptr.data(), indices.data(), data.data());
    dense_matrix<ScalarType> output = sm1 * dense;
    pybind11::gil_scoped_acquire acquire;
    return output;
}

#endif