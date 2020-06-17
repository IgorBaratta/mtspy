#ifdef USE_EIGEN_BACKEND

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

// declares a row-major dense matrix type of ScalarType
template <typename ScalarType>
using dense_matrix_t = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// declares a row-major sparse matrix type of ScalarType, using indices of type Indextype (int32 or int64)
template <typename ScalarType, typename IndType>
using sparse_matrix_t = Eigen::SparseMatrix<ScalarType, Eigen::RowMajor, IndType>;

template <typename ScalarType, typename IndType>
dense_matrix_t<ScalarType>
SpMM_eigen(IndType rows, IndType cols, IndType nnz,
           const Eigen::Ref<Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>> &data,
           const Eigen::Ref<Eigen::Matrix<IndType, Eigen::Dynamic, 1>> &indptr,
           const Eigen::Ref<Eigen::Matrix<IndType, Eigen::Dynamic, 1>> &indices,
           const Eigen::Ref<dense_matrix_t<ScalarType>> dense)
{
    // get data pointers
    const ScalarType *data_ptr = data.data();
    const IndType *displ_ptr = indptr.data();
    const IndType *indices_ptr = indices.data();

    pybind11::gil_scoped_release release;
    // Eigen only work with row-major sparse matrix in parallel
    Eigen::Map<const sparse_matrix_t<ScalarType, IndType>> sm1(rows, cols, nnz, displ_ptr, indices_ptr, data_ptr);
    dense_matrix_t<ScalarType> output = sm1 * dense;
    pybind11::gil_scoped_acquire acquire;
    return output;
}

#endif

bool has_eigen()
{
#ifdef USE_EIGEN_BACKEND
    return true;
#else
    return false;
#endif
}