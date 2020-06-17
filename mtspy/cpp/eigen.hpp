#ifdef USE_EIGEN_BACKEND

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

template <typename T>
using einge_array_t = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

// declares a row-major dense matrix type of ScalarType
template <typename ScalarType>
using eigen_dense_t = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// declares a row-major sparse matrix type of ScalarType in Eigen, using indices of
// type Indextype (int32 or int64).
// Note: Eigen only work with row-major sparse matrix in parallel
template <typename ScalarType, typename IndType>
using sparse_matrix_t = Eigen::SparseMatrix<ScalarType, Eigen::RowMajor, IndType>;

template <typename ScalarType, typename IndType>
eigen_dense_t<ScalarType>
sparse_dense_eigen(IndType rows, IndType cols, IndType nnz,
                   const einge_array_t<ScalarType> &data,
                   const einge_array_t<IndType> &indptr,
                   const einge_array_t<IndType> &indices,
                   const eigen_dense_t<ScalarType> &dense)
{
    // get data pointers
    const ScalarType *data_ptr = data.data();
    const IndType *displ_ptr = indptr.data();
    const IndType *indices_ptr = indices.data();

    // Temporarily  release global interpreter lock (GIL)
    pybind11::gil_scoped_release release;
    Eigen::Map<const sparse_matrix_t<ScalarType, IndType>> sm1(rows, cols, nnz, displ_ptr, indices_ptr, data_ptr);
    eigen_dense_t<ScalarType> output = sm1 * dense;

    pybind11::gil_scoped_acquire acquire;
    return output;
}

template <typename ScalarType, typename IndType>
sparse_matrix_t<ScalarType, IndType>
sparse_sparse_eigen(IndType A_rows, IndType A_cols, IndType A_nnz,
                    const einge_array_t<ScalarType> &A_data,
                    const einge_array_t<IndType> &A_indptr,
                    const einge_array_t<IndType> &A_indices,
                    IndType B_rows, IndType B_cols, IndType B_nnz,
                    const einge_array_t<ScalarType> &B_data,
                    const einge_array_t<IndType> &B_indptr,
                    const einge_array_t<IndType> &B_indices)
{

    // map matrix input matrices to eigen sparse matrix
    Eigen::Map<const sparse_matrix_t<ScalarType, IndType>> A_eigen(A_rows, A_cols, A_nnz,
                                                                   A_indptr.data(),
                                                                   A_indices.data(),
                                                                   A_data.data());

    Eigen::Map<const sparse_matrix_t<ScalarType, IndType>> B_eigen(B_rows, B_cols, B_nnz,
                                                                   B_indptr.data(),
                                                                   B_indices.data(),
                                                                   B_data.data());

    // Temporarily  release global interpreter lock (GIL)
    pybind11::gil_scoped_release release;
    sparse_matrix_t<ScalarType, IndType> C = A_eigen * B_eigen;
    pybind11::gil_scoped_acquire acquire;

    return C;
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